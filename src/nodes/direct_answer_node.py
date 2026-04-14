import time

from langchain_core.messages import HumanMessage

from src.agent.profile_utils import extract_preferred_topics
from src.agent.policy import (
    _looks_like_external_query,
    _looks_like_structured_db_query,
    is_complex_query,
    needs_rewrite_first,
)
from src.config.llm_config import LLMService
from src.models.llm import chatgpt_llm
from src.nodes.helpers import build_state_patch, create_event, finalize_event
from src.prompts.agent.direct_answer_prompt import DIRECT_ANSWER_PROMPT
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
    if state.long_term_memory_context and state.long_term_memory_context.strip():
        prompt = (
            f"{prompt}\n\n"
            "[长期记忆]\n"
            f"{state.long_term_memory_context.strip()}\n\n"
            "使用规则:\n"
            "- 这些长期记忆只用于帮助理解用户背景、偏好和长期任务。\n"
            "- 不要把它们当作实时事实来源。\n"
            "- 如果与用户当前问题冲突，以当前问题为准。"
        )
    return prompt


def _looks_like_unreliable_direct_answer(result: FinalAnswerResult) -> bool:
    answer = (result.answer or "").strip()
    lowered = answer.lower()
    uncertainty_markers = [
        "无法回答",
        "不能回答",
        "信息不足",
        "缺少信息",
        "不确定",
        "不清楚",
        "无法确定",
        "需要更多信息",
        "无法直接回答",
        "无法获取",
        "无法访问",
        "无法查询",
        "无法提供实时",
        "无法获取实时",
        "我无法获取",
        "请查看",
        "请查看您的设备",
        "请查看你的设备",
        "请以本地时间为准",
        "实时日期",
        "实时信息",
        "当前日期",
        "当前时间",
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
        "check your device",
        "check your local time",
        "real-time",
    ]

    if not answer:
        return True
    if result.fail_reason:
        return True
    if len(answer) <= 12:
        return True
    if any(marker in lowered or marker in answer for marker in uncertainty_markers):
        return True
    if "抱歉" in answer and ("无法" in answer or "不能" in answer):
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

    prompt = _build_direct_answer_prompt(state)

    try:
        result: FinalAnswerResult = LLMService.invoke(
            llm=chatgpt_llm,
            messages=[HumanMessage(content=prompt)],
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
