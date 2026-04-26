import time

from langchain_core.messages import HumanMessage

from src.agent.answer_stream import get_answer_token_handler
from src.agent.profile_utils import extract_preferred_topics
from src.config.llm_config import LLMService
from src.models.llm import chatgpt_llm
from src.nodes.helpers import build_state_patch, create_event, finalize_event
from src.prompts.agent.finalize_prompt import FINALIZE_PROMPT, FINALIZE_STREAM_PROMPT
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


def _append_long_term_memory_hint(state: State, prompt: str, *, stream_mode: bool) -> str:
    context = (state.long_term_memory_context or "").strip()
    if not context:
        return prompt

    instructions = (
        "Long-term memory can be used for user language/style preferences and other saved personal preferences.\n"
        "If the current evidence already supports the answer, do not let long-term memory change, extend, or override the evidence-based conclusion.\n"
        "Only use long-term memory content directly when the user explicitly asks about their own saved profile, preferences, identity, or history and the answer is not otherwise grounded in retrieved evidence.\n"
        "Do not use long-term memory as documentary evidence for enterprise facts, retrieved evidence, uploaded files, external facts, or real-time information.\n"
        "If it conflicts with current evidence, trust the current evidence.\n"
        "If it conflicts with the current chat context, trust the current chat context.\n"
        "Do not mention hidden memory systems.\n"
    )
    section_title = "[Long-term Memory Hint]" if stream_mode else "[Long-term Memory Context]"
    return f"{prompt}\n\n{section_title}\n{context}\n\n{instructions}"


def _build_finalize_prompt(
    state: State,
    *,
    effective_query: str,
    evidence_summary: str,
    citations: list[str],
) -> str:
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
    return _append_long_term_memory_hint(state, prompt, stream_mode=False)


def _build_finalize_stream_prompt(
    state: State,
    *,
    effective_query: str,
    evidence_summary: str,
) -> str:
    prompt = FINALIZE_STREAM_PROMPT.format(
        raw_query=state.query or "",
        query=effective_query,
        evidence_summary=evidence_summary or "No evidence summary available.",
        sub_query_context=_build_sub_query_context(state),
        output_level=state.output_level,
        preferred_language=_preferred_language(state),
        preferred_topics=", ".join(_preferred_topics(state)) or "None",
    ).strip()
    return _append_long_term_memory_hint(state, prompt, stream_mode=True)


def _build_fallback_final_answer(state: State) -> FinalAnswerResult:
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

    if evidence_summary:
        answer = evidence_summary
    elif fail_reason == "no_data":
        answer = "No reliable financial-report evidence was retrieved for this question."
    else:
        answer = "Current evidence is insufficient to produce a reliable answer."

    return FinalAnswerResult(
        success=bool(answer) and is_sufficient,
        answer=answer,
        citations=citations if _prefers_citations(state) else [],
        reason="finalize_fallback",
        fail_reason=fail_reason,
        diagnostics=["finalize_fallback_used"],
    )


def _invoke_finalize_result(
    state: State,
    *,
    effective_query: str,
    evidence_summary: str,
    citations: list[str],
    fail_reason: str | None,
    is_sufficient: bool,
) -> FinalAnswerResult:
    result: FinalAnswerResult = LLMService.invoke(
        llm=chatgpt_llm,
        messages=[
            HumanMessage(
                content=_build_finalize_prompt(
                    state,
                    effective_query=effective_query,
                    evidence_summary=evidence_summary,
                    citations=citations,
                )
            )
        ],
        schema=FinalAnswerResult,
    )
    result.citations = _normalize_citations(result.citations, citations)
    if not result.citations:
        result.citations = citations
    if result.fail_reason is None:
        result.fail_reason = fail_reason
    result.success = bool((result.answer or "").strip()) and is_sufficient
    if not result.reason:
        result.reason = "finalize_completed" if result.success else "finalize_constrained"
    if not result.diagnostics:
        result.diagnostics = ["finalize_llm_completed"]
    if state.long_term_memory_used:
        result.diagnostics.append("long_term_memory_hint_applied:finalize")
    if not _prefers_citations(state):
        result.citations = []
    return result


def _stream_finalize_result(
    state: State,
    *,
    effective_query: str,
    evidence_summary: str,
    citations: list[str],
    fail_reason: str | None,
    is_sufficient: bool,
) -> FinalAnswerResult:
    handler = get_answer_token_handler()
    if handler is None:
        raise RuntimeError("answer token handler is not bound")

    def on_token(token: str) -> None:
        handler(token)

    answer = LLMService.stream_text(
        llm=chatgpt_llm,
        messages=[
            HumanMessage(
                content=_build_finalize_stream_prompt(
                    state,
                    effective_query=effective_query,
                    evidence_summary=evidence_summary,
                )
            )
        ],
        on_token=on_token,
    ).strip()

    diagnostics = ["finalize_llm_stream_completed"]
    if state.long_term_memory_used:
        diagnostics.append("long_term_memory_hint_applied:finalize")

    return FinalAnswerResult(
        success=bool(answer) and is_sufficient,
        answer=answer,
        citations=citations if _prefers_citations(state) else [],
        reason="finalize_completed" if answer and is_sufficient else "finalize_constrained",
        fail_reason=fail_reason,
        diagnostics=diagnostics,
    )


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

    try:
        if get_answer_token_handler() is not None:
            try:
                result = _stream_finalize_result(
                    state,
                    effective_query=effective_query,
                    evidence_summary=evidence_summary,
                    citations=citations,
                    fail_reason=fail_reason,
                    is_sufficient=is_sufficient,
                )
            except Exception:
                result = _invoke_finalize_result(
                    state,
                    effective_query=effective_query,
                    evidence_summary=evidence_summary,
                    citations=citations,
                    fail_reason=fail_reason,
                    is_sufficient=is_sufficient,
                )
                result.diagnostics.append("finalize_stream_fallback_to_structured")
        else:
            result = _invoke_finalize_result(
                state,
                effective_query=effective_query,
                evidence_summary=evidence_summary,
                citations=citations,
                fail_reason=fail_reason,
                is_sufficient=is_sufficient,
            )
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
