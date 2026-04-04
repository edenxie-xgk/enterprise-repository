from langchain_core.messages import HumanMessage

from src.agent.router import reasoning_route, tool_route
from src.congfig.llm_config import LLMService
from src.models.llm import deepseek_llm
from src.prompts.agent.agent import AGENT_PROMPT
from src.types.agent_state import State
from src.types.base_type import BaseLLMDecideResult
from src.types.rag_type import RAGResult


def agent_node(state: State):
    """Agent 主节点逻辑，负责决定下一步动作

    Args:
        state: 当前状态对象，包含动作历史、查询等信息

    Returns:
        dict: 包含下一步动作和相关信息的字典，可能包含以下字段：
            - action: 下一步动作类型
            - answer: 最后获取的答案(如果是结束动作)
            - reason: 动作原因说明
            - status: 执行状态(success/failed)
            - fail_reason: 失败原因(如果失败)
    """
    # 获取最近一次工具调用记录
    last_tool = next((event for event in reversed(state.action_history) if event.kind == "tool"), None)

    # 检查是否超过最大步数限制
    if state.current_step >= state.max_steps:
        return {
            "action": "finish",
            "answer": get_last_answer(last_tool),
            "reason": f"Reached max steps: {state.max_steps}",
            "status": "failed",
            "fail_reason": "max_steps_exceeded",
            "diagnostics": state.diagnostics + ["agent:max_steps_exceeded"],
        }

    # 检查最近一次工具调用是否已返回足够结果
    last_event = state.action_history[-1] if state.action_history else None
    if last_event and last_event.kind == "tool" and getattr(last_event.output, "is_sufficient", False):
        answer = get_last_answer(last_tool)
        # 验证任务是否已完成
        if verify_task_complete(state, answer):
            return {
                "action": "finish",
                "answer": answer,
                "status": "success",
                "diagnostics": state.diagnostics + ["agent:task_complete"],
            }

    # 获取当前允许的动作列表
    allowed_actions = get_allowed_actions(state)
    # 构建Agent提示词
    prompt = AGENT_PROMPT.format(
        query=state.query,
        context=build_agent_context(state),
        query_evolution=build_query_evolution(state),
        allowed_actions=allowed_actions,
    )

    # 调用LLM服务决定下一步动作
    llm_result: BaseLLMDecideResult = LLMService.invoke(
        llm=deepseek_llm,
        messages=[HumanMessage(content=prompt)],
        schema=BaseLLMDecideResult,
    )

    # 处理LLM返回的动作
    action = llm_result.next_action
    # 如果动作不在允许列表中，则使用第一个允许的动作
    if action not in allowed_actions:
        action = allowed_actions[0]

    # 检查该动作是否已达到最大尝试次数
    if any(item.name == action and item.attempt >= item.max_attempt for item in reversed(state.action_history)):
        action = "finish"

    # 处理结束或中止动作
    if action in {"finish", "abort"}:
        return {
            "action": action,
            "answer": get_last_answer(last_tool),
            "reason": llm_result.reason,
            "status": "success" if action == "finish" else "failed",
            "fail_reason": None if action == "finish" else "tool_error",
            "diagnostics": state.diagnostics + [f"agent:{action}"],
        }

    # 返回继续执行的动作
    return {
        "action": action,
        "reason": llm_result.reason,
        "diagnostics": state.diagnostics + [f"agent:next={action}"],
    }


def get_last_answer(last_tool) -> str:
    if last_tool and last_tool.output:
        return getattr(last_tool.output, "answer", "") or ""
    return ""


def verify_task_complete(state: State, answer: str) -> bool:
    prompt = f"""
用户问题:
{state.query}

当前答案:
{answer}

请判断该答案是否已经完整回答用户问题。
只返回 true 或 false。
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


def get_allowed_actions(state: State) -> list[str]:
    if not state.action_history:
        return ["rag", "rewrite_query", "normalize_query"]

    last_tool = next((event for event in reversed(state.action_history) if event.kind == "tool"), None)
    if not last_tool:
        return ["rag"]

    trailing_reasoning = []
    for item in reversed(state.action_history):
        if item.kind == "reasoning":
            trailing_reasoning.append(item.name)
        else:
            break

    if trailing_reasoning:
        if len(trailing_reasoning) >= 2:
            return list(tool_route)
        actions = list(tool_route) + list(reasoning_route)
    else:
        fail_reason = getattr(last_tool.output, "fail_reason", None)
        if fail_reason == "no_data":
            actions = ["rewrite_query", "expand_query"]
        elif fail_reason == "low_recall":
            actions = ["expand_query", "rewrite_query", "decompose_query"]
        elif fail_reason == "ambiguous_query":
            actions = ["rewrite_query", "decompose_query", "expand_query"]
        elif fail_reason in {"bad_ranking", "bad_reranking"}:
            actions = ["decompose_query", "rewrite_query"]
        elif fail_reason == "verification_failed":
            actions = ["rag"]
        else:
            actions = ["rag"]

    actions = [action for action in actions if action not in trailing_reasoning] or ["rag"]
    if not trailing_reasoning:
        actions.extend(["finish", "abort"])
    return actions


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
