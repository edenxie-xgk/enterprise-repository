import time

from core.settings import settings
from src.memory import memory_service
from src.nodes.helpers import build_state_patch, create_event, finalize_event
from src.types.agent_state import State
from src.types.event_type import MemoryEvent
from src.types.memory_type import MemoryRecallQuery


def memory_recall_node(state: State):
    effective_query = state.resolved_query or state.working_query or state.query or ""
    start_time = time.time()

    event = create_event(
        MemoryEvent,
        name="memory_recall",
        input_data={
            "query": effective_query,
            "user_id": state.user_id or (state.user_profile or {}).get("user_id", ""),
            "session_id": state.session_id or "",
        },
        max_attempt=1,
    )
    event.attempt = 1

    recall_query = MemoryRecallQuery(
        user_id=str(state.user_id or (state.user_profile or {}).get("user_id") or ""),
        session_id=state.session_id or None,
        query=effective_query,
        top_k=settings.memory_top_k,
        min_score=settings.memory_recall_min_score,
        scopes=["user", "session"] if state.session_id else ["user"],
    )
    result = memory_service.recall(recall_query)
    event = finalize_event(event, result, start_time)

    return build_state_patch(
        state,
        event,
        long_term_memory_hits=result.memories,
        long_term_memory_context=result.memory_context or None,
        long_term_memory_used=result.used,
        last_memory_result=result,
    )
