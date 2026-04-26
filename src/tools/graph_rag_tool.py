from __future__ import annotations

from src.graph.service import graph_service
from src.types.graph_type import GraphQueryContext
from src.types.rag_type import RAGResult


def graph_rag_tool(context: GraphQueryContext, user_context: dict | None = None) -> RAGResult:
    """执行基于财报事实图谱的 Graph-RAG 查询。"""
    return graph_service.query(context, user_context or {})
