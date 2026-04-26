import time

from langchain_core.messages import HumanMessage

from src.agent.answer_stream import get_answer_token_handler
from src.agent.policy import (
    _looks_like_external_query,
    _looks_like_structured_db_query,
    is_complex_query,
    needs_rewrite_first,
)
from src.agent.profile_utils import extract_preferred_topics
from src.config.llm_config import LLMService
from src.models.llm import chatgpt_llm
from src.nodes.helpers import build_state_patch, create_event, finalize_event
from src.prompts.agent.direct_answer_prompt import (
    DIRECT_ANSWER_PROMPT,
    DIRECT_ANSWER_STREAM_PROMPT,
)
from src.types.agent_state import State
from src.types.event_type import ReasoningEvent
from src.types.final_answer_type import FinalAnswerResult


def _web_search_allowed(state: State) -> bool:
    profile = state.user_profile or {}
    return bool(profile.get("allow_web_search", False))


def _preferred_language(state: State) -> str:
    profile = state.user_profile or {}
    return str(profile.get("preferred_language") or "zh-CN").strip() or "zh-CN"


def _preferred_topics(state: State) -> list[str]:
    return extract_preferred_topics(state.user_profile or {})


def _append_long_term_memory_hint(state: State, prompt: str, *, stream_mode: bool) -> str:
    context = (state.long_term_memory_context or "").strip()
    if not context:
        return prompt

    instructions = (
        "Long-term memory can be used as authoritative context for user-provided profile facts, preferences, identity, and other saved personal facts.\n"
        "If the user asks about their own previously stated information, answer directly from long-term memory when it is relevant.\n"
        "Do not use long-term memory as documentary evidence for enterprise facts, uploaded files, external facts, or real-time information.\n"
        "If it conflicts with the current chat context, trust the current chat context.\n"
        "Do not mention hidden memory systems.\n"
    )
    section_title = "[Long-term Memory Hint]" if stream_mode else "[Long-term Memory Context]"
    return f"{prompt}\n\n{section_title}\n{context}\n\n{instructions}"


def _build_direct_answer_prompt(state: State) -> str:
    chat_history = "\n".join(state.chat_history[-8:]) if state.chat_history else "None"
    effective_query = state.working_query or state.resolved_query or state.query or ""
    prompt = DIRECT_ANSWER_PROMPT.format(
        raw_query=state.query or "",
        query=effective_query,
        chat_history=chat_history,
        output_level=state.output_level,
        preferred_language=_preferred_language(state),
        preferred_topics=", ".join(_preferred_topics(state)) or "None",
    ).strip()
    return _append_long_term_memory_hint(state, prompt, stream_mode=False)


def _build_direct_answer_stream_prompt(state: State) -> str:
    chat_history = "\n".join(state.chat_history[-8:]) if state.chat_history else "None"
    effective_query = state.working_query or state.resolved_query or state.query or ""
    prompt = DIRECT_ANSWER_STREAM_PROMPT.format(
        raw_query=state.query or "",
        query=effective_query,
        chat_history=chat_history,
        output_level=state.output_level,
        preferred_language=_preferred_language(state),
        preferred_topics=", ".join(_preferred_topics(state)) or "None",
    ).strip()
    return _append_long_term_memory_hint(state, prompt, stream_mode=True)


def _looks_like_unreliable_direct_answer(result: FinalAnswerResult) -> bool:
    answer = (result.answer or "").strip()
    lowered = answer.lower()
    uncertainty_markers = [
        "i cannot answer",
        "i can't answer",
        "i cannot access",
        "i cannot retrieve",
        "i do not have access",
        "i cannot provide real-time",
        "not enough information",
        "need more information",
        "insufficient information",
        "unclear",
        "not sure",
        "real-time",
        "\u65e0\u6cd5\u76f4\u63a5\u56de\u7b54",
        "\u65e0\u6cd5\u76f4\u63a5\u5b8c\u6210",
        "\u9700\u8981\u66f4\u591a\u4fe1\u606f",
        "\u4fe1\u606f\u4e0d\u8db3",
        "\u4e0a\u4e0b\u6587\u4e0d\u8db3",
        "\u65e0\u6cd5\u786e\u8ba4",
        "\u4e0d\u786e\u5b9a",
        "\u5b9e\u65f6",
        "\u5b9e\u65f6\u4fe1\u606f",
    ]

    if not answer:
        return True
    if result.fail_reason:
        return True
    if len(answer) <= 12:
        return True
    if any(marker in lowered or marker in answer for marker in uncertainty_markers):
        return True
    return False


def _select_fallback_action(state: State) -> tuple[str, str]:
    effective_query = (state.working_query or state.resolved_query or state.query or "").strip()
    if _looks_like_structured_db_query(effective_query):
        return "db_search", "direct_answer_failed_fallback_to_db_search"
    if _web_search_allowed(state) and _looks_like_external_query(effective_query):
        return "web_search", "direct_answer_failed_fallback_to_web_search"
    if needs_rewrite_first(effective_query):
        return "rewrite_query", "direct_answer_failed_fallback_to_rewrite"
    if is_complex_query(effective_query):
        return "decompose_query", "direct_answer_failed_fallback_to_decompose"
    return "rag", "direct_answer_failed_fallback_to_rag"


def _build_failed_result() -> FinalAnswerResult:
    return FinalAnswerResult(
        success=False,
        answer="",
        reason="direct_answer_unavailable",
        fail_reason="direct_answer_unavailable",
        diagnostics=["direct_answer_fallback_used"],
        message="direct answer unavailable; retry with tool",
    )


def _invoke_direct_answer_result(state: State) -> FinalAnswerResult:
    result: FinalAnswerResult = LLMService.invoke(
        llm=chatgpt_llm,
        messages=[HumanMessage(content=_build_direct_answer_prompt(state))],
        schema=FinalAnswerResult,
    )
    result.citations = []
    if not result.reason:
        result.reason = "direct_answer_completed"
    if not result.diagnostics:
        result.diagnostics = ["direct_answer_llm_completed"]
    result.diagnostics.append("direct_answer:standalone_node")
    if state.long_term_memory_used:
        result.diagnostics.append("long_term_memory_hint_applied:direct_answer")
    result.success = not _looks_like_unreliable_direct_answer(result)
    if not result.success and not result.fail_reason:
        result.fail_reason = "direct_answer_unavailable"
    return result


def _stream_direct_answer_result(state: State) -> FinalAnswerResult:
    handler = get_answer_token_handler()
    if handler is None:
        raise RuntimeError("answer token handler is not bound")

    def on_token(token: str) -> None:
        handler(token)

    answer = LLMService.stream_text(
        llm=chatgpt_llm,
        messages=[HumanMessage(content=_build_direct_answer_stream_prompt(state))],
        on_token=on_token,
    ).strip()

    result = FinalAnswerResult(
        success=True,
        answer=answer,
        citations=[],
        reason="direct_answer_completed",
        fail_reason=None,
        diagnostics=["direct_answer_llm_stream_completed", "direct_answer:standalone_node"],
    )
    if state.long_term_memory_used:
        result.diagnostics.append("long_term_memory_hint_applied:direct_answer")
    result.success = not _looks_like_unreliable_direct_answer(result)
    if not result.success and not result.fail_reason:
        result.fail_reason = "direct_answer_unavailable"
    return result


def direct_answer_node(state: State):
    effective_query = state.working_query or state.resolved_query or state.query or ""
    start_time = time.time()

    event = create_event(
        ReasoningEvent,
        name="direct_answer",
        input_data={
            "query": effective_query,
            "raw_query": state.query or "",
        },
        max_attempt=1,
    )
    event.attempt = 1

    try:
        result = _invoke_direct_answer_result(state)

        if result.success and get_answer_token_handler() is not None:
            try:
                streamed_result = _stream_direct_answer_result(state)
                if streamed_result.success:
                    result = streamed_result
                else:
                    result.diagnostics.append("direct_answer_stream_output_rejected_keep_structured")
            except Exception:
                result.diagnostics.append("direct_answer_stream_fallback_to_structured")
    except Exception:
        result = _build_failed_result()

    event = finalize_event(event, result, start_time)

    if not result.success:
        fallback_action, fallback_reason = _select_fallback_action(state)
        return build_state_patch(
            state,
            event,
            action=fallback_action,
            answer=state.answer,
            citations=state.citations,
            reason=fallback_reason,
            status="failed",
            fail_reason=result.fail_reason or "direct_answer_unavailable",
        )

    return build_state_patch(
        state,
        event,
        action="finish",
        answer=result.answer,
        citations=[],
        reason=result.reason or "direct_answer_completed",
        status="success",
        fail_reason=None,
    )
