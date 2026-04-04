import time

from src.nodes.helpers import build_state_patch, create_event, finalize_event, get_next_attempt
from src.models.llm import deepseek_llm
from src.tools.rewrite_query_tool import rewrite_query_tool, RewriteResult
from src.types.agent_state import State
from src.types.event_type import ReasoningEvent


def rewrite_query_node(state:State):
    start_time = time.time()
    query = state.working_query or state.query
    reasoning_event = create_event(
        ReasoningEvent,
        name="rewrite_query",
        input_data={"query": query},
    )
    if len(query) > 10:
        rewritten = rewrite_query_tool(deepseek_llm,state.working_query,state.chat_history)
    else:
        rewritten = RewriteResult(
            answer=query,
            success=True,
            tool_name="rewrite_query",
            message="重写为短查询跳过的查询",
            diagnostics=["rewrite_query_skipped_short_query"],
        )
    reasoning_event.attempt = get_next_attempt(state.action_history, "rewrite_query")
    reasoning_event = finalize_event(reasoning_event, rewritten, start_time)
    return build_state_patch(
        state,
        reasoning_event,
        working_query=rewritten.answer,
        rewrite_query=rewritten.answer,
    )
