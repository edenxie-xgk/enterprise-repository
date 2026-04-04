import time

from src.types.agent_state import State

from src.nodes.helpers import build_state_patch, create_event, finalize_event
from src.tools.normalize_query_tool import normalize_query
from src.types.base_type import BaseNodeResult
from src.types.event_type import ReasoningEvent


class NormalizeQueryResult(BaseNodeResult):
    pass

def normalize_query_node(state:State):
    start_time = time.time()
    query = normalize_query(state.query)
    event = create_event(
        ReasoningEvent,
        name="normalize_query",
        input_data={"query": state.query},
        max_attempt=1,
    )
    event.attempt = 1
    event = finalize_event(
        event,
        NormalizeQueryResult(answer=query, success=True),
        start_time,
    )
    return build_state_patch(
        state,
        event,
        normalized_query=query,
        working_query=query,
    )
