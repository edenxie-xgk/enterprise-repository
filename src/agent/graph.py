from langgraph.constants import END
from langgraph.graph import StateGraph

from src.nodes.agent_node import agent_node
from src.nodes.decompose_query_node import decompose_query_node
from src.nodes.db_search_node import db_search_node
from src.nodes.direct_answer_node import direct_answer_node
from src.nodes.expand_query_node import expand_query_node
from src.nodes.finalize_node import finalize_node
from src.nodes.graph_rag_node import graph_rag_node
from src.nodes.memory_recall_node import memory_recall_node
from src.nodes.rag_node import rag_node
from src.nodes.resolved_query_node import resolved_query_node
from src.nodes.rewrite_query_node import rewrite_query_node
from src.nodes.web_search_node import web_search_node
from src.agent.router import route_map
from src.types.agent_state import State

builder = StateGraph(State)


def route(state:State):
    # 动态跳转根据 Agent 决策输出
    return route_map.get(state.action, END)

builder.add_node("resolved_query", resolved_query_node)
builder.set_entry_point("resolved_query")
builder.add_node("memory_recall", memory_recall_node)

builder.add_node("agent", agent_node)

builder.add_node("direct_answer", direct_answer_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("expand_query", expand_query_node)
builder.add_node("decompose_query", decompose_query_node)
builder.add_node("rag", rag_node)
builder.add_node("graph_rag", graph_rag_node)
builder.add_node("web_search", web_search_node)
builder.add_node("db_search", db_search_node)
builder.add_node("finalize", finalize_node)

builder.add_edge("resolved_query", "memory_recall")
builder.add_edge("memory_recall", "agent")

builder.add_edge("rag", "agent")
builder.add_edge("graph_rag", "agent")
builder.add_edge("web_search", "agent")
builder.add_edge("db_search", "agent")
builder.add_conditional_edges(
    "direct_answer",
    route,
)
builder.add_edge("rewrite_query", "agent")
builder.add_edge("expand_query", "agent")
builder.add_edge("decompose_query", "agent")
builder.add_conditional_edges(
    "finalize",
    route,
)


builder.add_conditional_edges(
    'agent',
    route,
)

graph = builder.compile()


if __name__ == '__main__':
    print(graph.invoke(State(query="什么是金融里面包含什么样的知识，又怎么学？")))
