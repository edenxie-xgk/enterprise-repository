from typing import TypedDict, List, Dict, Optional

from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.models.llm import deepseek_llm
from src.tools.normalize_query_tool import normalize_query
from src.tools.rag_tool import rag_tool
from src.tools.rewrite_query_tool import rewrite_query_tool
from src.types.rag_type import RAGResult, RagContext


class State(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,  # 赋值时验证类型
        extra="forbid"  # 禁止额外字段，防止拼写错误
    )
    query: Optional[str]

    rewritten_query: Optional[str] = Field(default="",description="重写查询")

    expand_query:List[str] = Field(default=[],description="拓展查询")

    decompose_query:List[str] = Field(default=[],description="子任务规划")

    rag_context:Optional[RagContext] = Field(default_factory=lambda:RagContext(),description="rag请求上下文")

    rag_result:Optional[RAGResult] = Field(default_factory=lambda:RAGResult(),description="rag检索结果")

    answer:  Optional[str] = Field(default="",description="回答")

    chat_history:List[str] = Field(default=[],description="短期记忆")

    user_profile:Dict[str,any] = Field(default=None,description="用户画像")


llm = deepseek_llm


def normalize_query_node(state:State):
    query = normalize_query(state.query)
    state.rag_context.query = query
    return {"query":query,"rag_context":state.rag_context}

def rewrite_query_node(state:State):
    query = state.query
    if len(query) < 20:
        rewritten = rewrite_query_tool(llm,state.query)
    else:
        rewritten = state.query
    state.rag_context.rewritten_query = rewritten
    return {
        "rewritten_query": rewritten,
        "rag_context":state.rag_context
    }



def rag_node(state:State):
    print(state.rag_context)
    rag_result:RAGResult = rag_tool(state.rag_context,state.user_profile)
    return {"rag_result": rag_result}


def evaluate_node(state: State):
    result = state.rag_result

    if not result or len(result.documents) < 2:
        return {"reason": "low_recall"}
    return {"reason": "ok"}



builder = StateGraph(State)


builder.add_node("normalize_query", normalize_query_node)
builder.set_entry_point("normalize_query")

builder.add_node("rewrite_query", rewrite_query_node)
builder.add_edge("normalize_query", "rewrite_query")

builder.add_node("rag", rag_node)
builder.add_edge("rewrite_query", "rag")

builder.add_node("evaluate", evaluate_node)
builder.add_edge("rag", "evaluate")
builder.add_edge("evaluate", END)

graph = builder.compile()


if __name__ == '__main__':
    print(graph.invoke(State(query="什么是金融知识")))