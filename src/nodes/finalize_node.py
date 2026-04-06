import time

from langchain_core.messages import HumanMessage

from src.congfig.llm_config import LLMService
from src.models.llm import chatgpt_llm
from src.nodes.helpers import build_state_patch, create_event, finalize_event
from src.prompts.agent.finalize_prompt import FINALIZE_PROMPT
from src.types.agent_state import State
from src.types.event_type import ReasoningEvent
from src.types.final_answer_type import FinalAnswerResult


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

    return FinalAnswerResult(
        success=bool(answer) and is_sufficient,
        answer=answer,
        citations=citations,
        reason="finalize_fallback",
        fail_reason=fail_reason,
        diagnostics=["finalize_fallback_used"],
    )


def finalize_node(state: State):
    start_time = time.time()
    event = create_event(
        ReasoningEvent,
        name="finalize",
        input_data={
            "query": state.query,
            "evidence_summary": getattr(state.last_rag_result, "evidence_summary", ""),
        },
        max_attempt=1,
    )
    event.attempt = 1

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

    prompt = FINALIZE_PROMPT.format(
        query=state.query or "",
        evidence_summary=evidence_summary or "No evidence summary available.",
        sub_query_context=_build_sub_query_context(state),
    )

    try:
        result: FinalAnswerResult = LLMService.invoke(
            llm=chatgpt_llm,
            messages=[HumanMessage(content=prompt)],
            schema=FinalAnswerResult,
        )
        if not result.citations:
            result.citations = citations
        if result.fail_reason is None:
            result.fail_reason = fail_reason
        result.success = bool(result.answer) and is_sufficient
        if not result.reason:
            result.reason = "finalize_completed" if result.success else "finalize_constrained"
        if not result.diagnostics:
            result.diagnostics = ["finalize_llm_completed"]
    except Exception:
        result = _build_fallback_final_answer(state)

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
