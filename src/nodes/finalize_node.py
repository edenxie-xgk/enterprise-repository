import time

from langchain_core.messages import HumanMessage

from src.agent.profile_utils import extract_preferred_topics
from src.config.llm_config import LLMService
from src.models.llm import chatgpt_llm
from src.nodes.helpers import build_state_patch, create_event, finalize_event
from src.prompts.agent.finalize_prompt import FINALIZE_PROMPT
from src.types.agent_state import State
from src.types.event_type import ReasoningEvent
from src.types.final_answer_type import FinalAnswerResult


def _prefers_citations(state: State) -> bool:
    profile = state.user_profile or {}
    return bool(profile.get("prefers_citations", True))


def _preferred_language(state: State) -> str:
    profile = state.user_profile or {}
    return str(profile.get("preferred_language") or "zh-CN").strip() or "zh-CN"


def _preferred_topics(state: State) -> list[str]:
    return extract_preferred_topics(state.user_profile or {})


def _normalize_citations(raw_citations, allowed_citations: list[str]) -> list[str]:
    allowed_set = {item for item in allowed_citations if item}
    normalized: list[str] = []
    seen: set[str] = set()

    for item in raw_citations or []:
        citation = str(item).strip()
        if not citation:
            continue
        if citation not in allowed_set or citation in seen:
            continue
        seen.add(citation)
        normalized.append(citation)

    return normalized


def _build_sub_query_context(state: State) -> str:
    if not state.sub_query_results:
        return "No sub-query evidence."

    blocks = []
    for index, item in enumerate(state.sub_query_results, start=1):
        summary = getattr(item, "evidence_summary", "") or getattr(item, "answer", "")
        blocks.append(
            "\n".join(
                [
                    f"[Sub-query {index}]",
                    f"Question: {item.sub_query}",
                    f"Evidence summary: {summary}",
                    f"Is sufficient: {item.is_sufficient}",
                    f"Fail reason: {item.fail_reason or ''}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_fallback_final_answer(state: State) -> FinalAnswerResult:
    last_rag_result = state.last_rag_result
    evidence_summary = ""
    citations = []
    fail_reason = state.fail_reason
    is_sufficient = False

    if last_rag_result:
        evidence_summary = last_rag_result.evidence_summary or last_rag_result.answer or ""
        citations = list(last_rag_result.citations or [])
        fail_reason = last_rag_result.fail_reason or fail_reason
        is_sufficient = bool(last_rag_result.is_sufficient)

    if evidence_summary:
        answer = evidence_summary
    elif fail_reason == "no_data":
        answer = "当前知识库中未检索到足够相关内容，暂时无法给出可靠答案。"
    else:
        answer = "当前证据不足，暂时无法稳定回答该问题。"

    result = FinalAnswerResult(
        success=bool(answer) and is_sufficient,
        answer=answer,
        citations=citations,
        reason="finalize_fallback",
        fail_reason=fail_reason,
        diagnostics=["finalize_fallback_used"],
    )
    if not _prefers_citations(state):
        result.citations = []
    return result


def finalize_node(state: State):
    effective_query = state.working_query or state.resolved_query or state.query or ""
    start_time = time.time()

    event = create_event(
        ReasoningEvent,
        name="finalize",
        input_data={
            "query": effective_query,
            "evidence_summary": getattr(state.last_rag_result, "evidence_summary", ""),
        },
        max_attempt=1,
    )
    event.attempt = 1

    last_rag_result = state.last_rag_result
    evidence_summary = ""
    citations: list[str] = []
    fail_reason = state.fail_reason
    is_sufficient = False

    if last_rag_result:
        evidence_summary = last_rag_result.evidence_summary or last_rag_result.answer or ""
        citations = list(last_rag_result.citations or [])
        fail_reason = last_rag_result.fail_reason or fail_reason
        is_sufficient = bool(last_rag_result.is_sufficient)

    prompt = FINALIZE_PROMPT.format(
        raw_query=state.query or "",
        query=effective_query,
        evidence_summary=evidence_summary or "No evidence summary available.",
        sub_query_context=_build_sub_query_context(state),
        available_citations=", ".join(citations) if citations else "None",
        output_level=state.output_level,
        preferred_language=_preferred_language(state),
        prefers_citations="yes" if _prefers_citations(state) else "no",
        preferred_topics=", ".join(_preferred_topics(state)) or "None",
    )
    if state.long_term_memory_context and state.long_term_memory_context.strip():
        prompt = (
            f"{prompt}\n\n"
            "[长期记忆]\n"
            f"{state.long_term_memory_context.strip()}\n\n"
            "使用规则:\n"
            "- 长期记忆只用于帮助保持跨会话的一致语气、背景和偏好。\n"
            "- 不要让长期记忆覆盖证据摘要里的事实。\n"
            "- 如果长期记忆与证据冲突，以证据为准。"
        )

    try:
        result: FinalAnswerResult = LLMService.invoke(
            llm=chatgpt_llm,
            messages=[HumanMessage(content=prompt)],
            schema=FinalAnswerResult,
        )
        result.citations = _normalize_citations(result.citations, citations)
        if not result.citations:
            result.citations = citations
        if result.fail_reason is None:
            result.fail_reason = fail_reason
        result.success = bool(result.answer) and is_sufficient
        if not result.reason:
            result.reason = "finalize_completed" if result.success else "finalize_constrained"
        if not result.diagnostics:
            result.diagnostics = ["finalize_llm_completed"]
        if state.long_term_memory_used:
            result.diagnostics.append("long_term_memory_hint_applied:finalize")
    except Exception:
        result = _build_fallback_final_answer(state)

    if not _prefers_citations(state):
        result.citations = []

    event = finalize_event(event, result, start_time)

    return build_state_patch(
        state,
        event,
        action="finish",
        answer=result.answer,
        citations=result.citations,
        reason=result.reason or "",
        status="success" if result.success else "failed",
        fail_reason=result.fail_reason,
    )
