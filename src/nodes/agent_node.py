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


def agent_node(state: State):
    """Agent决策主节点函数

    根据当前状态决定下一步动作，处理各种边界条件和决策逻辑

    Args:
        state: 当前Agent状态对象，包含查询、历史动作等信息

    Returns:
        dict: 包含下一步动作和相关信息的字典，结构为:
            - action: 下一步动作类型
            - answer: 当前答案(如果有)
            - reason: 决策原因
            - status: 执行状态(success/failed)
            - fail_reason: 失败原因(如果失败)
            - diagnostics: 诊断信息列表
    """
    # 获取最近一次工具调用事件
    last_tool = next((event for event in reversed(state.action_history) if event.kind == "tool"), None)

    # 输入安全检查
    input_guard = guard_input(state.query or state.working_query or "")
    if not input_guard.is_valid:
        return {
            "action": "finish",
            "answer": input_guard.response,
            "reason": input_guard.reason,
            "status": "failed",
            "fail_reason": "disallowed_query" if input_guard.reason in DISALLOWED_INPUT_REASONS else "invalid_input",
            "diagnostics": state.diagnostics + [f"agent:input_blocked:{input_guard.reason}"],
        }

    # 最大步数检查
    if state.current_step >= state.max_steps:
        return {
            "action": "finish",
            "answer": get_last_answer(last_tool),
            "reason": f"Reached max steps: {state.max_steps}",
            "status": "failed",
            "fail_reason": "max_steps_exceeded",
            "diagnostics": state.diagnostics + ["agent:max_steps_exceeded"],
        }

    # 检查最近工具调用是否已提供足够答案
    last_event = state.action_history[-1] if state.action_history else None
    if last_event and last_event.kind == "tool" and getattr(last_event.output, "is_sufficient", False):
        answer = get_last_answer(last_tool)
        if verify_task_complete(state, answer):
            return {
                "action": "finish",
                "answer": answer,
                "status": "success",
                "diagnostics": state.diagnostics + ["agent:task_complete"],
            }

    force_finish, finish_reason = should_force_finish(state)
    if force_finish:
        return {
            "action": "finish",
            "answer": get_last_answer(last_tool) or build_fallback_answer(state),
            "reason": finish_reason,
            "status": "failed",
            "fail_reason": getattr(state.last_rag_result, "fail_reason", None) or "insufficient_context",
            "diagnostics": state.diagnostics + [f"agent:force_finish:{finish_reason}"],
        }

    # 初始决策逻辑(当没有推理历史时)
    reasoning_history = [event for event in state.action_history if event.kind == "reasoning"]
    if not reasoning_history:
        initial_decision = decide_initial_action(state)
        if initial_decision.next_action == "clarify_question":
            return {
                "action": "finish",
                "answer": initial_decision.clarification_question or "请补充你的问题背景、对象或范围。",
                "reason": initial_decision.reason,
                "status": "success",
                "fail_reason": "ambiguous_query",
                "diagnostics": state.diagnostics + ["agent:clarify_question"],
            }

        return {
            "action": initial_decision.next_action,
            "reason": initial_decision.reason,
            "diagnostics": state.diagnostics + [f"agent:initial={initial_decision.next_action}"],
        }

    # 准备LLM决策所需的上下文和提示
    allowed_actions = get_allowed_actions(state)
    prompt = AGENT_PROMPT.format(
        query=state.query,
        context=build_agent_context(state),
        query_evolution=build_query_evolution(state),
        allowed_actions=allowed_actions,
    )

    # 调用LLM进行决策
    llm_result: BaseLLMDecideResult = LLMService.invoke(
        llm=deepseek_llm,
        messages=[HumanMessage(content=prompt)],
        schema=BaseLLMDecideResult,
    )

    # 处理LLM返回的决策结果
    if llm_result is None:
        return {
            "action": allowed_actions[0],
            "reason": "agent_llm_returned_none_fallback",
            "diagnostics": state.diagnostics + ["agent:llm_none_fallback"],
        }

    action = llm_result.next_action or allowed_actions[0]
    # 确保动作在允许范围内
    if action not in allowed_actions:
        action = allowed_actions[0]

    # 检查动作尝试次数是否超过限制
    if any(item.name == action and item.attempt >= item.max_attempt for item in reversed(state.action_history)):
        action = "finish"

    # 处理终止类动作(finish/abort)
    if action in {"finish", "abort"}:
        return {
            "action": action,
            "answer": get_last_answer(last_tool),
            "reason": llm_result.reason,
            "status": "success" if action == "finish" else "failed",
            "fail_reason": None if action == "finish" else "tool_error",
            "diagnostics": state.diagnostics + [f"agent:{action}"],
        }

    # 返回常规动作决策
    return {
        "action": action,
        "reason": llm_result.reason,
        "diagnostics": state.diagnostics + [f"agent:next={action}"],
    }


def get_last_answer(last_tool) -> str:
    if last_tool and last_tool.output:
        return getattr(last_tool.output, "answer", "") or ""
    return ""


def build_fallback_answer(state: State) -> str:
    if state.sub_query_results:
        return "已尝试拆解并检索多个子问题，但当前证据仍不足以稳定回答原问题。"
    if state.last_rag_result and state.last_rag_result.fail_reason == "no_data":
        return "当前知识库中未检索到足够相关内容，暂时无法给出可靠答案。"
    return "当前信息不足，暂时无法稳定回答该问题。"


def verify_task_complete(state: State, answer: str) -> bool:
    prompt = f"""
用户问题:
{state.query}

当前答案:
{answer}

请判断该答案是否已经完整回答用户问题。只返回 true 或 false。
"""
    llm_result = LLMService.invoke(
        llm=deepseek_llm,
        messages=[HumanMessage(content=prompt)],
    )
    content = getattr(llm_result, "content", llm_result)
    if isinstance(content, str):
        return content.strip().lower() == "true"
    if isinstance(content, bool):
        return content
    return False


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
            quality_hint = {"has_data": len(result.documents) > 0}
            answer = result.answer
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
                    f"Answer: {answer}",
                    f"Quality hint: {quality_hint}",
                    f"Is sufficient: {is_sufficient}",
                    f"Fail reason: {fail_reason}",
                ]
            )
        )

    if not lines:
        return "No tool executions yet."
    return "\n\n".join(lines)
