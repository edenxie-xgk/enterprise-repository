import time

from src.nodes.helpers import build_state_patch, create_event, finalize_event, get_next_attempt
from src.models.llm import deepseek_llm
from src.tools.expand_query_tool import expand_query_tool
from src.types.agent_state import State
from src.types.event_type import ReasoningEvent


def expand_query_node(state:State):
    start_time = time.time()
    event = create_event(
        ReasoningEvent,
        name="expand_query",
        input_data={"query":state.working_query or  state.query},
        max_attempt=1,
    )
    expand_query = expand_query_tool(deepseek_llm,state.working_query,state.chat_history)
    event.attempt = get_next_attempt(state.action_history, "expand_query")
    event = finalize_event(event, expand_query, start_time)
    return build_state_patch(
        state,
        event,
        working_query=state.working_query,
        expand_query=expand_query.answer,
    )
