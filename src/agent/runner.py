from __future__ import annotations

from typing import Any
from uuid import uuid4

from src.agent.graph import graph
from src.types.agent_state import State


def run_agent(
    query: str,
    *,
    user_id: str = "",
    session_id: str = "",
    user_profile: dict[str, Any] | None = None,
    chat_history: list[str] | None = None,
    max_steps: int = 6,
) -> State:
    """运行智能体处理用户查询

    初始化智能体状态并执行处理流程，返回最终状态对象

    Args:
        query: 用户输入的查询内容
        user_id: 用户ID，默认为空字符串
        session_id: 会话ID，默认为空字符串
        user_profile: 用户画像信息字典，可选
        chat_history: 聊天历史记录列表，可选
        max_steps: 最大执行步骤数，默认为6

    Returns:
        State: 包含智能体运行结果的状态对象
    """
    # 初始化状态对象，生成唯一run_id并设置各属性
    initial_state = State(
        query=query,
        run_id=str(uuid4()),  # 使用uuid生成唯一运行ID
        user_id=user_id,
        session_id=session_id,
        user_profile=user_profile,
        chat_history=chat_history or [],  # 确保chat_history不为None
        max_steps=max_steps,
    )

    # 调用图处理流程执行智能体逻辑
    result = graph.invoke(initial_state)

    # 返回处理结果，确保返回的是State对象
    if isinstance(result, State):
        return result
    return State(**result)  # 如果结果是字典则转换为State对象


def summarize_trace(state: State) -> list[dict[str, Any]]:
    """将智能体运行轨迹(trace)信息转换为字典列表形式

    遍历State对象中的trace信息，将每个跟踪项转换为字典格式，
    便于后续处理或序列化为JSON等格式

    Args:
        state: 包含智能体运行轨迹信息的State对象

    Returns:
        list[dict[str, Any]]: 包含所有跟踪项信息的字典列表，
                            每个字典包含以下键：
                            - step: 步骤序号
                            - event: 事件名称
                            - kind: 事件类型
                            - status: 执行状态
                            - attempt: 尝试次数
                            - duration_ms: 执行时长(毫秒)
                            - fail_reason: 失败原因(如有)
                            - message: 相关信息
                            - diagnostics: 诊断信息
    """
    rows = []
    # 遍历state中的每个trace项
    for item in state.trace:
        # 将每个trace项转换为字典格式
        rows.append(
            {
                "step": item.step,  # 当前执行步骤序号
                "event": item.event_name,  # 发生的事件名称
                "kind": item.event_kind,  # 事件类型分类
                "status": item.status,  # 执行状态(成功/失败等)
                "attempt": item.attempt,  # 当前尝试次数
                "duration_ms": item.duration_ms,  # 该步骤耗时(毫秒)
                "fail_reason": item.fail_reason,  # 失败原因(如果失败)
                "message": item.message,  # 相关消息或描述
                "diagnostics": item.diagnostics,  # 诊断信息或详细数据
            }
        )
    return rows


def build_run_report(state: State) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "query": state.query,
        "status": state.status,
        "action": state.action,
        "answer": state.answer,
        "reason": state.reason,
        "fail_reason": state.fail_reason,
        "current_step": state.current_step,
        "max_steps": state.max_steps,
        "diagnostics": state.diagnostics,
        "trace": summarize_trace(state),
    }


if __name__ == "__main__":
    final_state = run_agent("什么是金融里面包含什么样的知识，又该怎么学？")
    report = build_run_report(final_state)
    print(report)
