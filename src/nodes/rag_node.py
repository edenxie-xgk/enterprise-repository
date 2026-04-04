import time

from src.nodes.helpers import build_state_patch, create_event, finalize_event, get_next_attempt
from src.types.agent_state import State
from src.tools.rag_tool import rag_tool
from src.types.event_type import ToolEvent
from src.types.rag_type import RAGResult, RagContext


def rag_node(state:State):
    start_time = time.time()

    last_rag = next((item for item  in state.action_history[::-1] if item.name == 'rag'),None)
    new_tool = create_event(
        ToolEvent,
        name="rag",
    )
    if last_rag: #拿上一次的调用+agent建议调整来请求（这里agent调整暂时不补充）
        new_input = RagContext(
            **last_rag.input.dict(),
        )
    else:
        new_input = RagContext(
        )
    new_input.query = state.query
    new_input.rewritten_query =  state.rewrite_query
    new_input.expand_query =  state.expand_query
    new_input.decompose_query =  state.decompose_query

    tool_result:RAGResult = rag_tool(RagContext(**new_input.dict()),state.user_profile)

    new_tool.attempt = get_next_attempt(state.action_history, "rag")

    new_tool.input = new_input
    new_tool = finalize_event(new_tool, tool_result, start_time)

    return build_state_patch(
        state,
        new_tool,
        last_rag_context=new_input,
        last_rag_result=tool_result,
    )
