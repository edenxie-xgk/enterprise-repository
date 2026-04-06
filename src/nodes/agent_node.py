from langchain_core.messages import HumanMessage

from src.agent.policy import decide_initial_action, get_allowed_actions, guard_input, should_force_finish
from src.congfig.llm_config import LLMService
from src.models.llm import deepseek_llm
from src.prompts.agent.agent import AGENT_PROMPT
from src.types.agent_state import State
from src.types.base_type import BaseLLMDecideResult
from src.types.rag_type import RAGResult


DISALLOWED_INPUT_REASONS = {
    "illegal_cyber_activity",
    "privacy_exfiltration",
    "illegal_deception_request",
}


def _has_finalize_material(state: State) -> bool:
    last_rag_result = state.last_rag_result
    if last_rag_result:
        if last_rag_result.documents:
            return True
        if last_rag_result.evidence_summary or last_rag_result.answer:
            return True
    return bool(state.sub_query_results)


def _can_finalize(state: State, *, allow_partial: bool = False) -> bool:
    """判断当前状态是否可以完成最终响应

    根据最后一次RAG检索结果的质量指标，判断是否满足完成条件

    Args:
        state: 包含当前agent状态的对象
        allow_partial: 是否允许部分完成（当证据不足但上下文不完整时）

    Returns:
        bool: 是否可以完成最终响应
    """
    # 获取最后一次RAG检索结果
    last_rag_result = state.last_rag_result
    if not last_rag_result:
        return False

    # 检查关键质量指标
    has_summary = bool((last_rag_result.evidence_summary or last_rag_result.answer or "").strip())  # 是否有摘要或答案
    has_citations = bool(last_rag_result.citations)  # 是否有引用
    has_documents = bool(last_rag_result.documents)  # 是否有文档
    is_sufficient = bool(last_rag_result.is_sufficient)  # 结果是否足够充分
    fail_reason = getattr(last_rag_result, "fail_reason", None)  # 可能的失败原因

    # 完全满足条件的情况：结果充分且有摘要、引用和文档
    if is_sufficient and has_summary and has_citations and has_documents:
        return True

    # 如果不允许部分完成，则直接返回False
    if not allow_partial:
        return False

    # 允许部分完成的条件：有摘要且(有引用或文档)且失败原因为上下文不足
    return (
        has_summary
        and (has_citations or has_documents)
        and fail_reason == "insufficient_context"
    )


def _build_finish_response(state: State, *, answer: str, reason: str = "", status: str = "success", fail_reason=None, diagnostics=None):
    return {
        "action": "finish",
        "answer": answer,
        "reason": reason,
        "status": status,
        "fail_reason": fail_reason,
        "diagnostics": diagnostics or state.diagnostics,
    }


def agent_node(state: State):
    """Agent 决策主节点，负责根据当前状态决定下一步动作

    Args:
        state: State - 包含当前 agent 状态的对象，包括查询、历史动作、诊断信息等

    Returns:
        dict: 包含以下键的字典：
            - action: 下一步动作类型 (如 "finalize", "finish" 等)
            - reason: 动作决策的原因说明
            - diagnostics: 诊断信息列表
            或包含更多信息的完成响应字典（当返回 _build_finish_response 时）
    """
    # 获取最近的工具调用事件和最后一个事件
    last_tool = next((event for event in reversed(state.action_history) if event.kind == "tool"), None)
    last_event = state.action_history[-1] if state.action_history else None

    # 输入安全检查
    input_guard = guard_input(state.query or state.working_query or "")
    if not input_guard.is_valid:
        return _build_finish_response(
            state,
            answer=input_guard.response,
            reason=input_guard.reason,
            status="failed",
            fail_reason="disallowed_query" if input_guard.reason in DISALLOWED_INPUT_REASONS else "invalid_input",
            diagnostics=state.diagnostics + [f"agent:input_blocked:{input_guard.reason}"],
        )

    # 检查是否超过最大步骤限制
    if state.current_step >= state.max_steps:
        if _can_finalize(state, allow_partial=True):
            return {
                "action": "finalize",
                "reason": f"Reached max steps: {state.max_steps}",
                "diagnostics": state.diagnostics + ["agent:max_steps_finalize"],
            }
        return _build_finish_response(
            state,
            answer=get_last_answer(last_tool) or build_fallback_answer(state),
            reason=f"Reached max steps: {state.max_steps}",
            status="failed",
            fail_reason="max_steps_exceeded",
            diagnostics=state.diagnostics + ["agent:max_steps_exceeded"],
        )

    # 检查是否有足够的检索证据可以完成
    if (
        last_event
        and last_event.kind == "tool"
        and getattr(last_event.output, "is_sufficient", False)
        and _can_finalize(state, allow_partial=False)
    ):
        return {
            "action": "finalize",
            "reason": "retrieval_evidence_is_sufficient",
            "diagnostics": state.diagnostics + ["agent:finalize_after_sufficient_evidence"],
        }

    # 检查是否需要强制结束
    force_finish, finish_reason = should_force_finish(state)
    if force_finish:
        if _can_finalize(state, allow_partial=True):
            return {
                "action": "finalize",
                "reason": finish_reason,
                "diagnostics": state.diagnostics + [f"agent:force_finalize:{finish_reason}"],
            }
        return _build_finish_response(
            state,
            answer=get_last_answer(last_tool) or build_fallback_answer(state),
            reason=finish_reason or "force_finish",
            status="failed",
            fail_reason=getattr(state.last_rag_result, "fail_reason", None) or "insufficient_context",
            diagnostics=state.diagnostics + [f"agent:force_finish:{finish_reason}"],
        )

    # 如果没有推理历史，进行初始决策
    reasoning_history = [event for event in state.action_history if event.kind == "reasoning"]
    if not reasoning_history:
        initial_decision = decide_initial_action(state)
        if initial_decision.next_action == "clarify_question":
            return _build_finish_response(
                state,
                answer=initial_decision.clarification_question or "请补充你的问题背景、对象或范围。",
                reason=initial_decision.reason or "clarify_question",
                status="success",
                fail_reason="ambiguous_query",
                diagnostics=state.diagnostics + ["agent:clarify_question"],
            )

        return {
            "action": initial_decision.next_action,
            "reason": initial_decision.reason,
            "diagnostics": state.diagnostics + [f"agent:initial={initial_decision.next_action}"],
        }

    # 构建 LLM 提示并调用决策
    allowed_actions = get_allowed_actions(state)
    prompt = AGENT_PROMPT.format(
        query=state.query,
        context=build_agent_context(state),
        query_evolution=build_query_evolution(state),
        allowed_actions=allowed_actions,
    )

    llm_result: BaseLLMDecideResult = LLMService.invoke(
        llm=deepseek_llm,
        messages=[HumanMessage(content=prompt)],
        schema=BaseLLMDecideResult,
    )

    # LLM 返回 None 时的后备处理
    if llm_result is None:
        fallback_action = allowed_actions[0]
        if fallback_action == "finish" and _can_finalize(state, allow_partial=True):
            fallback_action = "finalize"
        return {
            "action": fallback_action,
            "reason": "agent_llm_returned_none_fallback",
            "diagnostics": state.diagnostics + ["agent:llm_none_fallback"],
        }

    # 处理 LLM 返回的动作决策
    action = llm_result.next_action or allowed_actions[0]
    if action not in allowed_actions:
        action = allowed_actions[0]

    # 检查动作尝试次数是否超过限制
    if any(item.name == action and item.attempt >= item.max_attempt for item in reversed(state.action_history)):
        action = "finalize" if _can_finalize(state, allow_partial=True) else "finish"

    # 处理 finish 动作
    if action == "finish":
        if _can_finalize(state, allow_partial=True):
            return {
                "action": "finalize",
                "reason": llm_result.reason or "llm_requested_finish_with_evidence",
                "diagnostics": state.diagnostics + ["agent:finalize_before_finish"],
            }
        return _build_finish_response(
            state,
            answer=get_last_answer(last_tool) or build_fallback_answer(state),
            reason=llm_result.reason or "finish",
            status="success",
            diagnostics=state.diagnostics + ["agent:finish"],
        )

    # 处理 abort 动作
    if action == "abort":
        return _build_finish_response(
            state,
            answer=get_last_answer(last_tool) or build_fallback_answer(state),
            reason=llm_result.reason or "abort",
            status="failed",
            fail_reason="tool_error",
            diagnostics=state.diagnostics + ["agent:abort"],
        )

    # 返回常规动作决策
    return {
        "action": action,
        "reason": llm_result.reason,
        "diagnostics": state.diagnostics + [f"agent:next={action}"],
    }


def get_last_answer(last_tool) -> str:
    if last_tool and last_tool.output:
        return getattr(last_tool.output, "evidence_summary", None) or getattr(last_tool.output, "answer", "") or ""
    return ""


def build_fallback_answer(state: State) -> str:
    if state.sub_query_results:
        return "已尝试拆解并检索多个子问题，但当前证据仍不足以稳定回答原问题。"
    if state.last_rag_result and state.last_rag_result.fail_reason == "no_data":
        return "当前知识库中未检索到足够相关内容，暂时无法给出可靠答案。"
    return "当前信息不足，暂时无法稳定回答该问题。"


def build_query_evolution(state: State) -> str:
    reasoning_events = [event for event in state.action_history if event.kind == "reasoning"]
    if not reasoning_events:
        return "No reasoning steps yet."

    lines = []
    for index, event in enumerate(reasoning_events[-3:], start=1):
        source_query = event.input.get("query") if isinstance(event.input, dict) else ""
        answer = getattr(event.output, "answer", "")
        target = "|".join(answer) if isinstance(answer, list) else answer
        lines.append(f"Q{index}: {source_query} -> {target} ({event.name})")

    lines.append(f"Current query: {state.working_query}")
    return "\n".join(lines)


def build_agent_context(state: State, last_tool_num: int = 3) -> str:
    lines = []
    if state.working_memory:
        lines.append(f"[Working memory]\n{state.working_memory}")

    event_list = [event for event in state.action_history if event.kind == "tool"][-last_tool_num:]
    for index, event in enumerate(event_list, start=1):
        if event.name == "rag":
            result: RAGResult = event.output
            quality_hint = {"has_data": len(result.documents) > 0, "confidence": result.confidence}
            answer = result.evidence_summary or result.answer
            fail_reason = result.fail_reason
            is_sufficient = result.is_sufficient
        else:
            quality_hint = {"has_data": True}
            answer = str(event.output)
            fail_reason = ""
            is_sufficient = False

        lines.append(
            "\n".join(
                [
                    f"[Recent tool {index}]",
                    f"Tool: {event.name}",
                    f"Input: {event.input}",
                    f"Evidence summary: {answer}",
                    f"Quality hint: {quality_hint}",
                    f"Is sufficient: {is_sufficient}",
                    f"Fail reason: {fail_reason}",
                ]
            )
        )

    if not lines:
        return "No tool executions yet."
    return "\n\n".join(lines)
