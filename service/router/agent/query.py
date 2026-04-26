import asyncio
from typing import Any, Literal

from fastapi import Depends
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
from service.utils.user_profile import (
    build_user_profile_payload,
    get_or_create_user_profile,
    sync_user_profile_from_query,
)
from src.agent.runner import build_run_report, run_agent


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    session_id: str = Field(..., description="Chat session id")
    output_level: Literal["concise", "standard", "detailed"] | None = Field(
        default=None,
        description="Answer verbosity: concise|standard|detailed",
    )

@agent_router.post("/query")
async def query_agent(
    payload: AgentQueryRequest,
    current_user: UserModel = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
):
    """处理代理查询请求

    Args:
        payload: 包含查询参数、会话ID和输出级别的请求体
        current_user: 通过认证的当前用户对象
        session: 异步数据库会话

    Returns:
        dict: 包含最终状态和运行报告的字典
    """
    # 查询当前用户角色对应的部门ID列表
    role_dept_result = await session.execute(
        select(RoleDepartmentModel.dept_id).where(RoleDepartmentModel.role_id == current_user.role_id)
    )
    allowed_department_ids = role_dept_result.scalars().all()

    # 构建用户档案信息
    profile = await get_or_create_user_profile(session=session, current_user=current_user)
    user_profile = build_user_profile_payload(
        current_user=current_user,
        allowed_department_ids=allowed_department_ids,
        profile=profile,
    )

    # 初始化聊天历史记录
    chat_history: list[str] = []
    if payload.session_id:
        # 异步获取指定会话的最近聊天历史
        chat_history = await asyncio.to_thread(
            chat_store.get_recent_history,
            session_id=payload.session_id,
            user_id=int(current_user.id),
            limit=settings.agent_chat_history_limit,
        )

    # 运行代理处理查询
    final_state = run_agent(
        payload.query,
        user_id=str(current_user.id or ""),
        session_id=payload.session_id,
        chat_history=chat_history,
        user_profile=user_profile,
        max_steps=settings.agent_max_steps,
        output_level=payload.output_level or user_profile.get("answer_style"),
    )
    _, synced_user_profile, profile_sync_summary = await sync_user_profile_from_query(
        session=session,
        current_user=current_user,
        allowed_department_ids=allowed_department_ids,
        profile=profile,
        query=payload.query,
    )
    final_state.user_profile = synced_user_profile
    if profile_sync_summary is not None:
        final_state.memory_write_summary = dict(final_state.memory_write_summary or {})
        final_state.memory_write_summary["profile_sync"] = profile_sync_summary

    # 返回处理结果
    return {
        "status": final_state.status,
        "data": build_run_report(final_state),
    }
