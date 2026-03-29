from typing import Dict, Any

from src.tools.rewrite_query_tool import RewriteResult
from src.types.base_type import ToolEvent
from src.types.rag_type import RAGResult, RagContext


def build_agent_context(event) -> Dict[str, Any]:
    """
    将不同 tool 的 output 转成统一格式
    """

    if event.tool_name == "rag":
        result:RAGResult = event.output  # RAGResult
        query:RagContext  = event.input  # RAGResult
        return {
            "tool_name": "rag",
            "query":query.rewritten_query or  query.query,
            "answer": result.answer,
            "evidence": [
                getattr(doc, "content", "") for doc in result.documents[:5]
            ],
            "fail_reason": result.fail_reason,
            "quality_hint": {
                "has_data": len(result.documents) > 0,
                "is_sufficient": result.is_sufficient,
            }
        }

    # 未来扩展
    elif event.tool_name == "web":
        result = event.output
        return {
            "tool_name": "web",
            "answer": result.answer,
            "evidence": result.snippets[:5],
            "quality_hint": {
                "has_data": len(result.snippets) > 0,
                "is_sufficient": result.is_sufficient,
            }
        }

    elif event.tool_name == "db":
        result = event.output
        return {
            "tool_name": "db",
            "answer": str(result.rows),
            "evidence": [str(result.rows[:5])],
            "quality_hint": {
                "has_data": len(result.rows) > 0,
                "is_sufficient": result.is_sufficient,
            }
        }

    # fallback（非常重要）
    return {
        "tool_name": event.tool_name,
        "query": "",
        "answer": str(event.output),
        "fail_reason":"",
        "evidence": [],
        "quality_hint": {
            "has_data": True,
            "is_sufficient": False,
        }
    }