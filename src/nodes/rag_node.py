import time
from typing import Any

from src.agent.policy import build_retrieval_plan
from src.nodes.helpers import build_state_patch, create_event, finalize_event, get_next_attempt
from src.tools.rag_tool import rag_tool
from src.types.agent_state import State
from src.types.event_type import ToolEvent
from src.types.rag_type import RAGResult, RagContext


def _normalize_filter_value(value: Any):
    """规范化过滤值

    将输入的过滤值转换为标准格式，处理各种边界情况：
    1. 空值处理：None或空字符串转为None
    2. 字符串处理：去除首尾空格，数字字符串转为整数
    3. 集合类型处理：递归处理每个元素，去重并简化单元素集合
    4. 其他类型：原样返回

    Args:
        value: 输入的过滤值，可以是任意类型

    Returns:
        规范化后的值，可能为None、int、str或集合类型
    """
    # 处理空值和空字符串情况
    if value is None or value == "":
        return None

    # 字符串类型处理
    if isinstance(value, str):
        normalized = value.strip()
        # 处理空字符串
        if not normalized:
            return None
        # 数字字符串转为整数
        if normalized.isdigit():
            return int(normalized)
        return normalized

    # 处理集合类型（列表、元组、集合）
    if isinstance(value, (list, tuple, set)):
        normalized_items = []
        seen = set()  # 用于去重的标记集合
        # 递归处理每个元素
        for item in value:
            normalized_item = _normalize_filter_value(item)
            if normalized_item is None:
                continue
            # 使用repr作为唯一标记进行去重
            marker = repr(normalized_item)
            if marker in seen:
                continue
            seen.add(marker)
            normalized_items.append(normalized_item)

        # 处理空集合情况
        if not normalized_items:
            return None
        # 单元素集合简化为单个值
        if len(normalized_items) == 1:
            return normalized_items[0]
        return normalized_items

    # 其他类型直接返回
    return value


def _first_profile_value(profile: dict[str, Any], *keys: str):
    for key in keys:
        if key in profile:
            return profile.get(key)
    return None


def build_access_filters(state: State) -> tuple[dict[str, Any], list[str], bool]:
    profile = state.user_profile or {}
    filters: dict[str, Any] = {}
    diagnostics: list[str] = []

    allowed_department_ids = _first_profile_value(
        profile,
        "allowed_department_ids",
        "department_ids",
        "dept_ids",
    )
    department_id = _first_profile_value(profile, "department_id", "dept_id")
    allowed_user_ids = _first_profile_value(profile, "allowed_user_ids", "user_ids")
    profile_user_id = _first_profile_value(profile, "user_id")

    normalized_department_ids = _normalize_filter_value(allowed_department_ids)
    normalized_department_id = _normalize_filter_value(department_id)
    normalized_user_ids = _normalize_filter_value(allowed_user_ids)
    normalized_user_id = _normalize_filter_value(profile_user_id or state.user_id)

    if allowed_department_ids is not None:
        if normalized_department_ids is None:
            diagnostics.append("access_filter:empty_allowed_department_ids")
            return {}, diagnostics, True
        filters["department_id"] = normalized_department_ids
        diagnostics.append(f"access_filter:department_id={normalized_department_ids}")
    elif normalized_department_id is not None:
        filters["department_id"] = normalized_department_id
        diagnostics.append(f"access_filter:department_id={normalized_department_id}")

    if allowed_user_ids is not None:
        if normalized_user_ids is None:
            diagnostics.append("access_filter:empty_allowed_user_ids")
            return {}, diagnostics, True
        filters["user_id"] = normalized_user_ids
        diagnostics.append(f"access_filter:user_id={normalized_user_ids}")
    elif "department_id" not in filters and normalized_user_id is not None:
        filters["user_id"] = normalized_user_id
        diagnostics.append(f"access_filter:user_id={normalized_user_id}")

    return filters, diagnostics, False


def rag_node(state: State):
    """RAG检索节点主函数

    负责处理RAG(Retrieval-Augmented Generation)检索流程，包括：
    1. 构建检索上下文
    2. 处理访问权限控制
    3. 执行检索计划
    4. 返回检索结果

    Args:
        state: 当前状态对象，包含用户查询、历史记录等信息

    Returns:
        返回更新后的状态补丁，包含检索结果和上下文
    """
    # 记录开始时间用于性能统计
    start_time = time.time()

    # 从历史记录中查找最近的RAG操作
    last_rag = next((item for item in state.action_history[::-1] if item.name == "rag"), None)
    # 创建新的工具事件
    new_tool = create_event(ToolEvent, name="rag")

    # 初始化检索上下文：如果有历史记录则复用，否则创建新上下文
    if last_rag:
        new_input = RagContext(**last_rag.input.dict())
    else:
        new_input = RagContext()

    # 填充查询相关字段
    new_input.query = state.query
    new_input.rewritten_query = state.rewrite_query
    new_input.expand_query = state.expand_query
    new_input.decompose_query = state.decompose_query

    # 构建访问过滤器并合并到现有过滤条件
    access_filters, access_diagnostics, access_denied = build_access_filters(state)
    merged_filters = dict(new_input.filters or {})
    merged_filters.update(access_filters)
    new_input.filters = merged_filters

    # 处理访问被拒绝的情况
    if access_denied:
        tool_result = RAGResult(
            success=False,
            tool_name="rag",
            answer="当前用户没有可用的数据访问范围，无法执行检索。",
            is_sufficient=False,
            fail_reason="permission_denied",
            retrieval_queries=[],
            diagnostics=access_diagnostics,
        )
        # 设置尝试次数并完成事件
        new_tool.attempt = get_next_attempt(state.action_history, "rag")
        new_tool.input = new_input
        new_tool = finalize_event(new_tool, tool_result, start_time)
        # 返回拒绝访问的状态补丁
        return build_state_patch(
            state,
            new_tool,
            last_rag_context=new_input,
            last_rag_result=tool_result,
            sub_query_results=[],
        )

    # 构建检索计划并更新上下文参数
    retrieval_plan = build_retrieval_plan(state, new_input)
    new_input.retrieval_top_k = retrieval_plan.retrieval_top_k
    new_input.rerank_top_k = retrieval_plan.rerank_top_k
    new_input.use_retrieval = retrieval_plan.use_retrieval
    new_input.use_rerank = retrieval_plan.use_rerank

    # 调用RAG工具执行实际检索
    tool_result: RAGResult = rag_tool(
        RagContext(**new_input.dict()),
        state.user_profile,
        previous_result=state.last_rag_result,
    )
    # 添加诊断信息
    tool_result.diagnostics.append(f"policy:{retrieval_plan.strategy_reason}")
    tool_result.diagnostics.extend(access_diagnostics)

    # 完成工具事件设置
    new_tool.attempt = get_next_attempt(state.action_history, "rag")
    new_tool.input = new_input
    new_tool = finalize_event(new_tool, tool_result, start_time)

    # 返回包含检索结果的状态补丁
    return build_state_patch(
        state,
        new_tool,
        last_rag_context=new_input,
        last_rag_result=tool_result,
        sub_query_results=tool_result.metadata.get("sub_query_results", []),
    )
