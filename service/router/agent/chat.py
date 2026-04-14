from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Literal
from uuid import uuid4

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from core.settings import settings
from service.database.connect import get_session
from service.dependencies.auth import get_current_active_user
from service.models.role_department import RoleDepartmentModel
from service.models.users import UserModel
from service.router.agent.index import agent_router
from service.utils.chat_store import chat_store
from service.utils.user_profile import build_user_profile_payload, get_or_create_user_profile
from src.agent.runner import build_run_report, run_agent


class ChatStreamRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    session_id: str | None = Field(default=None, description="Existing session id")
    output_level: Literal["concise", "standard", "detailed"] | None = Field(
        default=None,
        description="Answer verbosity: concise|standard|detailed",
    )

def _chunk_text(text: str, chunk_size: int = 16) -> list[str]:
    value = text or ""
    if not value:
        return []
    return [value[i : i + chunk_size] for i in range(0, len(value), chunk_size)]


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _build_allowed_department_ids(
    *,
    current_user: UserModel,
    session: AsyncSession,
) -> list[int]:
    role_dept_result = await session.execute(
        select(RoleDepartmentModel.dept_id).where(RoleDepartmentModel.role_id == current_user.role_id)
    )
    return role_dept_result.scalars().all()


@agent_router.get("/sessions")
async def list_chat_sessions(current_user: UserModel = Depends(get_current_active_user)):
    sessions = await asyncio.to_thread(chat_store.list_sessions, user_id=int(current_user.id))
    return {"code": 200, "message": "success", "data": sessions}


@agent_router.get("/sessions/{session_id}/messages")
async def get_chat_session_messages(
    session_id: str,
    current_user: UserModel = Depends(get_current_active_user),
):
    session_doc = await asyncio.to_thread(
        chat_store.get_session,
        session_id=session_id,
        user_id=int(current_user.id),
    )
    if not session_doc:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await asyncio.to_thread(
        chat_store.list_messages,
        session_id=session_id,
        user_id=int(current_user.id),
    )
    run_map = await asyncio.to_thread(
        chat_store.get_runs_by_ids,
        run_ids=[item.get("run_id") for item in messages if item.get("run_id")],
    )
    enriched_messages = []
    for message in messages:
        report = (run_map.get(message.get("run_id")) or {}).get("report") or {}
        enriched_messages.append(
            {
                **message,
                "report_summary": {
                    "status": report.get("status"),
                    "fail_reason": report.get("fail_reason"),
                    "action": report.get("action"),
                    "reason": report.get("reason"),
                    "output_level": report.get("output_level"),
                    "user_profile": report.get("user_profile"),
                    "preferred_topics_usage": report.get("preferred_topics_usage"),
                    "long_term_memory_used": report.get("long_term_memory_used"),
                    "long_term_memory_context": report.get("long_term_memory_context"),
                    "memory_write_summary": report.get("memory_write_summary") or {},
                    "citation_details": report.get("citation_details") or [],
                    "trace": report.get("trace") or [],
                    "action_history": report.get("action_history") or [],
                }
                if report
                else None,
            }
        )
    return {"code": 200, "message": "success", "data": {"session": session_doc, "messages": enriched_messages}}


@agent_router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: UserModel = Depends(get_current_active_user),
):
    deleted = await asyncio.to_thread(
        chat_store.soft_delete_session,
        session_id=session_id,
        user_id=int(current_user.id),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"code": 200, "message": "success", "data": {"session_id": session_id}}


@agent_router.post("/chat/stream")
async def stream_chat(
    payload: ChatStreamRequest,
    current_user: UserModel = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    allowed_department_ids = await _build_allowed_department_ids(current_user=current_user, session=session)
    profile = await get_or_create_user_profile(session=session, current_user=current_user)
    user_profile = build_user_profile_payload(
        current_user=current_user,
        allowed_department_ids=allowed_department_ids,
        profile=profile,
    )
    current_session_id = payload.session_id

    if current_session_id:
        session_doc = await asyncio.to_thread(
            chat_store.get_session,
            session_id=current_session_id,
            user_id=int(current_user.id),
        )
        if not session_doc:
            raise HTTPException(status_code=404, detail="Session not found")

    async def event_stream() -> AsyncIterator[str]:
        nonlocal current_session_id
        try:
            if not current_session_id:
                session_doc = await asyncio.to_thread(
                    chat_store.create_session,
                    user_id=int(current_user.id),
                    first_query=query,
                )
                current_session_id = session_doc["session_id"]
                yield _format_sse("session_created", session_doc)

            chat_history = await asyncio.to_thread(
                chat_store.get_recent_history,
                session_id=current_session_id,
                user_id=int(current_user.id),
                limit=settings.agent_chat_history_limit,
            )

            await asyncio.to_thread(
                chat_store.create_message,
                session_id=current_session_id,
                user_id=int(current_user.id),
                role="user",
                content=query,
            )

            assistant_message_id = str(uuid4())
            yield _format_sse(
                "message_started",
                {
                    "session_id": current_session_id,
                    "role": "assistant",
                    "message_id": assistant_message_id,
                    "output_level": payload.output_level or user_profile.get("answer_style") or settings.agent_output_level,
                },
            )

            final_state = await asyncio.to_thread(
                run_agent,
                query,
                user_id=str(current_user.id or ""),
                session_id=current_session_id,
                chat_history=chat_history,
                user_profile=user_profile,
                max_steps=settings.agent_max_steps,
                output_level=payload.output_level or user_profile.get("answer_style"),
            )
            report = build_run_report(final_state)
            answer = (report.get("answer") or "").strip()
            citations = report.get("citations") or []
            if not answer:
                answer = report.get("reason") or "未生成有效回答。"

            draft_message = await asyncio.to_thread(
                chat_store.create_message,
                session_id=current_session_id,
                user_id=int(current_user.id),
                role="assistant",
                content=answer,
                citations=citations,
                message_id=assistant_message_id,
            )

            run_doc = await asyncio.to_thread(
                chat_store.create_run,
                session_id=current_session_id,
                user_id=int(current_user.id),
                message_id=assistant_message_id,
                query=query,
                report=report,
            )
            await asyncio.to_thread(
                chat_store.attach_run_id,
                message_id=assistant_message_id,
                user_id=int(current_user.id),
                run_id=run_doc["run_id"],
            )
            draft_message["run_id"] = run_doc["run_id"]

            for chunk in _chunk_text(answer):
                yield _format_sse(
                    "token",
                    {
                        "session_id": current_session_id,
                        "message_id": assistant_message_id,
                        "content": chunk,
                    },
                )
                await asyncio.sleep(0.01)

            yield _format_sse(
                "message_completed",
                {
                    "session_id": current_session_id,
                    "message": draft_message,
                    "report_summary": {
                        "status": report.get("status"),
                        "fail_reason": report.get("fail_reason"),
                        "action": report.get("action"),
                        "reason": report.get("reason"),
                        "output_level": report.get("output_level"),
                        "user_profile": report.get("user_profile"),
                        "preferred_topics_usage": report.get("preferred_topics_usage"),
                        "long_term_memory_used": report.get("long_term_memory_used"),
                        "long_term_memory_context": report.get("long_term_memory_context"),
                        "memory_write_summary": report.get("memory_write_summary") or {},
                        "citations": citations,
                        "citation_details": report.get("citation_details") or [],
                        "trace": report.get("trace") or [],
                        "action_history": report.get("action_history") or [],
                    },
                },
            )
        except Exception as exc:
            yield _format_sse(
                "error",
                {
                    "session_id": current_session_id,
                    "message": str(exc),
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
