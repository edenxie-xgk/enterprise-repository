from __future__ import annotations

import argparse
import json
import time
from typing import Any
from uuid import uuid4

from core.settings import settings
from src.memory.writeback import write_long_term_memory
from src.config.llm_config import LLMService
from src.types.agent_state import State
from src.types.memory_type import MemoryWriteRequest


def _build_citation_label(document: Any) -> str:
    metadata = getattr(document, "metadata", None)
    if metadata is None and isinstance(document, dict):
        metadata = document.get("metadata") or {}

    def pick(name: str, default=None):
        if metadata is None:
            return default
        if isinstance(metadata, dict):
            return metadata.get(name, default)
        return getattr(metadata, name, default)

    file_name = pick("file_name", "") or "未知文档"
    page = pick("page", None)
    sheet_name = pick("sheet_name", "") or ""
    # section_title = pick("section_title", "") or ""
    # chunk_index = pick("chunk_index", None)

    extras = []
    if page not in (None, "", 0):
        extras.append(f"第{page}页")
    if sheet_name:
        extras.append(f"Sheet:{sheet_name}")
    # if section_title:
    #     extras.append(section_title)
    # if chunk_index not in (None, "", 0):
    #     extras.append(f"片段{chunk_index}")

    if extras:
        return f"{file_name} ({' | '.join(str(item) for item in extras)})"
    return file_name


def _build_citation_details(last_rag_result: Any) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """从RAG结果中构建引用详情信息

    处理RAG结果中的引用信息，生成三种形式的引用数据：
    1. 原始引用ID列表
    2. 可读的引用标签列表
    3. 详细的引用元数据字典列表

    Args:
        last_rag_result: RAG处理结果对象，应包含citations和documents属性

    Returns:
        tuple: 包含三个元素的元组：
            - raw_citations: 原始引用ID列表
            - citation_labels: 可读的引用标签列表
            - citation_details: 包含详细元数据的引用字典列表
    """
    # 获取原始引用列表和文档列表，确保返回的是列表类型
    raw_citations = list(getattr(last_rag_result, "citations", []) or [])
    documents = list(getattr(last_rag_result, "documents", []) or [])

    # 构建文档映射表，以node_id为键
    doc_map: dict[str, Any] = {}
    for doc in documents:
        node_id = getattr(doc, "node_id", None)
        if node_id and node_id not in doc_map:
            doc_map[node_id] = doc

    # 初始化返回结果容器
    citation_labels: list[str] = []
    citation_details: list[dict[str, Any]] = []

    # 处理每个引用，生成标签和详情信息
    for citation in raw_citations:
        # 获取对应的文档对象
        doc = doc_map.get(citation)
        # 生成可读的引用标签
        label = _build_citation_label(doc) if doc else str(citation)

        if label not in citation_labels:
            citation_labels.append(label)

        # 构建基础详情信息
        detail = {
            "node_id": citation,
            "label": label,
        }

        # 如果文档存在，添加元数据信息
        if doc:
            metadata = getattr(doc, "metadata", None)
            if hasattr(metadata, "model_dump"):  # 处理Pydantic模型
                detail["metadata"] = metadata.model_dump()
            elif isinstance(metadata, dict):  # 处理普通字典
                detail["metadata"] = metadata
        citation_details.append(detail)

    return raw_citations, citation_labels, citation_details


def run_agent(
    query: str,
    *,
    user_id: str = "",
    session_id: str = "",
    user_profile: dict[str, Any] | None = None,
    chat_history: list[str] | None = None,
    max_steps: int = 20,
    output_level: str | None = None,
) -> State:
    """Run the agent graph and return the final state."""
    from src.agent.graph import graph

    resolved_output_level = (
        output_level
        or (user_profile or {}).get("answer_style")
        or settings.agent_output_level
    )
    initial_state = State(
        query=query,
        run_id=str(uuid4()),
        user_id=user_id,
        session_id=session_id,
        user_profile=user_profile,
        chat_history=chat_history or [],
        max_steps=max_steps,
        output_level=resolved_output_level,
    )
    usage_token = LLMService.start_usage_collection()
    start = time.time()
    try:
        result = graph.invoke(initial_state)
    finally:
        llm_records = LLMService.stop_usage_collection(usage_token)
    duration_ms = int((time.time() - start) * 1000)
    if isinstance(result, State):
        state = result
    else:
        state = State(**result)
    state.duration_ms = duration_ms
    state.llm_usage = LLMService.summarize_usage(llm_records)
    try:
        memory_write_result = write_long_term_memory(
            MemoryWriteRequest(
                user_id=state.user_id or "",
                session_id=state.session_id or None,
                query=state.query or "",
                answer=state.answer or "",
                chat_history=state.chat_history or [],
                user_profile=state.user_profile or {},
                existing_memories=state.long_term_memory_hits,
            )
        )
        state.memory_write_summary = {
            "written_count": memory_write_result.written_count,
            "skipped_count": memory_write_result.skipped_count,
            "memory_ids": memory_write_result.memory_ids,
            "diagnostics": memory_write_result.diagnostics,
        }
    except Exception as exc:
        state.memory_write_summary = {
            "written_count": 0,
            "skipped_count": 0,
            "memory_ids": [],
            "diagnostics": [f"memory_write_exception={exc}"],
        }
    return state


def summarize_trace(state: State) -> list[dict[str, Any]]:
    """Convert trace records into a JSON-friendly list."""
    rows = []
    for item in state.trace:
        rows.append(
            {
                "step": item.step,
                "event": item.event_name,
                "kind": item.event_kind,
                "status": item.status,
                "attempt": item.attempt,
                "duration_ms": item.duration_ms,
                "fail_reason": item.fail_reason,
                "message": item.message,
                "diagnostics": item.diagnostics,
            }
        )
    return rows


def _summarize_event_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    elif isinstance(payload, dict):
        data = dict(payload)
    else:
        return str(payload)

    if not isinstance(data, dict):
        return data

    compact: dict[str, Any] = {}
    for key in [
        "query",
        "rewritten_query",
        "expand_query",
        "decompose_query",
        "answer",
        "message",
        "reason",
        "fail_reason",
        "is_sufficient",
        "citations",
        "retrieval_queries",
        "retrieval_candidate_node_ids",
        "rerank_node_ids",
        "documents",
        "filters",
        "use_retrieval",
        "use_rerank",
        "retrieval_top_k",
        "rerank_top_k",
    ]:
        value = data.get(key)
        if value not in (None, "", [], {}):
            compact[key] = value

    if "documents" in compact and isinstance(compact["documents"], list):
        compact["documents"] = len(compact["documents"])
    return compact or data


def summarize_action_history(state: State) -> list[dict[str, Any]]:
    rows = []
    for item in state.action_history:
        rows.append(
            {
                "id": item.id,
                "kind": item.kind,
                "name": item.name,
                "status": item.status,
                "attempt": item.attempt,
                "max_attempt": item.max_attempt,
                "started_at": item.started_at,
                "ended_at": item.ended_at,
                "duration_ms": item.duration,
                "error": item.error,
                "input": _summarize_event_payload(item.input),
                "output": _summarize_event_payload(item.output),
            }
        )
    return rows


def _summarize_user_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not profile:
        return None

    return {
        "user_id": profile.get("user_id"),
        "username": profile.get("username"),
        "department_id": profile.get("department_id") or profile.get("dept_id"),
        "role_id": profile.get("role_id"),
        "allowed_department_ids": list(profile.get("allowed_department_ids") or []),
        "answer_style": profile.get("answer_style"),
        "preferred_language": profile.get("preferred_language"),
        "preferred_topics": list(profile.get("preferred_topics") or []),
        "prefers_citations": bool(profile.get("prefers_citations", True)),
        "allow_web_search": bool(profile.get("allow_web_search", False)),
    }


def _extract_preferred_topics_usage(state: State) -> dict[str, Any]:
    profile = state.user_profile or {}
    available_topics = list(profile.get("preferred_topics") or [])
    applied_steps: list[str] = []
    guidance_query_count = 0
    matched_diagnostics: list[str] = []

    for item in state.action_history:
        output = getattr(item, "output", None)
        diagnostics = list(getattr(output, "diagnostics", []) or [])
        topic_diagnostics = [
            diag for diag in diagnostics
            if isinstance(diag, str) and ("preferred_topics" in diag or "preferred_topic" in diag)
        ]
        if not topic_diagnostics:
            continue

        if item.name not in applied_steps:
            applied_steps.append(item.name)
        matched_diagnostics.extend(topic_diagnostics)

        for diag in topic_diagnostics:
            if diag.startswith("preferred_topic_guidance_queries="):
                _, _, raw_value = diag.partition("=")
                try:
                    guidance_query_count = max(guidance_query_count, int(raw_value or "0"))
                except ValueError:
                    continue

    return {
        "available_topics": available_topics,
        "used": bool(applied_steps),
        "applied_steps": applied_steps,
        "guidance_query_count": guidance_query_count,
        "matched_diagnostics": matched_diagnostics,
    }


def build_run_report(state: State) -> dict[str, Any]:
    last_rag_result = state.last_rag_result
    raw_citations, citation_labels, citation_details = _build_citation_details(last_rag_result) if last_rag_result else ([], [], [])
    sub_query_results = []
    for item in state.sub_query_results:
        if hasattr(item, "model_dump"):
            sub_query_results.append(item.model_dump())
        elif isinstance(item, dict):
            sub_query_results.append(item)

    return {
        "run_id": state.run_id,
        "query": state.query,
        "resolved_query": state.resolved_query,
        "working_query": state.working_query,
        "status": state.status,
        "output_level": state.output_level,
        "action": state.action,
        "answer": state.answer,
        "citations": citation_labels,
        "raw_citations": raw_citations,
        "citation_details": citation_details,
        "reason": state.reason,
        "fail_reason": state.fail_reason,
        "current_step": state.current_step,
        "max_steps": state.max_steps,
        "duration_ms": state.duration_ms,
        "llm_usage": state.llm_usage,
        "working_memory": state.working_memory,
        "short_term_memory": state.short_term_memory,
        "long_term_memory_used": state.long_term_memory_used,
        "long_term_memory_context": state.long_term_memory_context,
        "long_term_memory_hits": [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in (state.long_term_memory_hits or [])
        ],
        "memory_write_summary": state.memory_write_summary,
        "user_profile": _summarize_user_profile(state.user_profile),
        "preferred_topics_usage": _extract_preferred_topics_usage(state),
        "diagnostics": state.diagnostics,
        "rewrite_query": state.rewrite_query,
        "expand_query": state.expand_query,
        "decompose_query": state.decompose_query,
        "sub_query_results": sub_query_results,
        "action_history": summarize_action_history(state),
        "last_rag_result": {
            "answer": getattr(last_rag_result, "answer", ""),
            "evidence_summary": getattr(last_rag_result, "evidence_summary", ""),
            "is_sufficient": getattr(last_rag_result, "is_sufficient", False),
            "fail_reason": getattr(last_rag_result, "fail_reason", None),
            "confidence": getattr(last_rag_result, "confidence", None),
            "citations": citation_labels,
            "raw_citations": raw_citations,
            "citation_details": citation_details,
            "retrieval_queries": getattr(last_rag_result, "retrieval_queries", []),
            "retrieval_candidate_node_ids": getattr(last_rag_result, "retrieval_candidate_node_ids", []),
            "rerank_node_ids": getattr(last_rag_result, "rerank_node_ids", []),
            "diagnostics": getattr(last_rag_result, "diagnostics", []),
            "document_count": len(getattr(last_rag_result, "documents", []) or []),
        }
        if last_rag_result
        else None,
        "trace": summarize_trace(state),
    }


def print_run_report(state: State) -> None:
    print(json.dumps(build_run_report(state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the Agentic RAG graph for local validation.")
    # parser.add_argument(
    #     "query",
    #     default="根据多个会议记录，不同部门对新能源行业景气度的看法有哪些共同点和差异？",
    #     help="User query",
    # )
    # parser.add_argument("--user-id", default="", help="User ID")
    # parser.add_argument("--session-id", default="", help="Session ID")
    # parser.add_argument("--max-steps", type=int, default=6, help="Maximum agent steps")
    # args = parser.parse_args()

    final_state = run_agent(
        query="python怎么学？",
        # user_id=args.user_id,
        # session_id=args.session_id,
        # max_steps=args.max_steps,
    )
    print_run_report(final_state)
