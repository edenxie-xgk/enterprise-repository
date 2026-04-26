import time

from src.graph.service import graph_service
from src.nodes.helpers import build_state_patch, create_event, finalize_event, get_next_attempt
from src.nodes.rag_node import build_access_filters
from src.tools.graph_rag_tool import graph_rag_tool
from src.types.agent_state import State
from src.types.event_type import ToolEvent
from src.types.graph_type import GraphQueryContext
from src.types.rag_type import RAGResult


def graph_rag_node(state: State):
    effective_query = state.working_query or state.resolved_query or state.query or ""
    start_time = time.time()
    new_tool = create_event(ToolEvent, name="graph_rag")

    graph_context = graph_service.build_query_context(
        query=effective_query,
        rewritten_query=state.rewrite_query,
        expand_query=state.expand_query,
        decompose_query=state.decompose_query,
        filters={},
    )

    access_filters, access_diagnostics, access_denied = build_access_filters(state)
    graph_context.filters.update(access_filters)

    if access_denied:
        tool_result = RAGResult(
            success=False,
            name="graph_rag",
            answer="当前用户没有可用的数据访问范围，无法执行图谱检索。",
            is_sufficient=False,
            fail_reason="permission_denied",
            retrieval_queries=[],
            diagnostics=access_diagnostics,
        )
        new_tool.attempt = get_next_attempt(state.action_history, "graph_rag")
        new_tool.input = graph_context
        new_tool = finalize_event(new_tool, tool_result, start_time)
        return build_state_patch(
            state,
            new_tool,
            last_graph_context=graph_context,
            last_graph_result=tool_result,
            last_rag_result=tool_result,
        )

    tool_result: RAGResult = graph_rag_tool(GraphQueryContext(**graph_context.model_dump()), state.user_profile)
    tool_result.diagnostics.extend(access_diagnostics)

    new_tool.attempt = get_next_attempt(state.action_history, "graph_rag")
    new_tool.input = graph_context
    new_tool = finalize_event(new_tool, tool_result, start_time)

    return build_state_patch(
        state,
        new_tool,
        last_graph_context=graph_context,
        last_graph_result=tool_result,
        last_rag_result=tool_result,
    )
