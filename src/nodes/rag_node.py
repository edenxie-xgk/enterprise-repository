import copy
from src.types.agent_state import State
from src.tools.rag_tool import rag_tool
from src.types.base_type import  ToolEvent
from src.types.rag_type import RAGResult, RagContext


# class DecodeRagResult(BaseLLMDecideResult):
#     """Agent 决策系统返回的数据"""
#     fail_reason: Literal[
#         "low_recall",
#         "bad_ranking",
#         "ambiguous_query",
#         "no_data",
#     ]  = Field(default=None,description="诊断信息")
#     suggested_actions:List[Literal[
#         "retry",
#         "rewrite",
#         "expand",
#         "decompose",
#         "retrieval",
#         "rerank",
#         "abort"
#     ]] = Field(default=None,description="行为建议")

def rag_node(state:State):

    last_rag = next((item for item  in state.tool_history[::-1] if item.tool_name == 'rag'),None)
    new_tool = ToolEvent(
        tool_name="rag",
    )
    if last_rag: #拿上一次的调用+agent建议调整来请求（这里agent调整暂时不补充）
        new_input = RagContext(
            **last_rag.input.dict(),
        )
    else:
        new_input = RagContext(
            query=state.query
        )
    if state.working_query != state.query:
        new_input.rewritten_query =  state.working_query or new_input.rewritten_query
    else:
        new_input.rewritten_query = None

    tool_result:RAGResult = rag_tool(RagContext(**new_input.dict()),state.user_profile)

    if last_rag:
        tool_result.attempt = last_rag.output.attempt + 1
    else:
        tool_result.attempt = 1

    new_tool.input = new_input
    new_tool.output = tool_result

    return {
        "answer":tool_result.answer,
        "tool_history":state.tool_history + [new_tool],
        "phase": "observe"
    }
