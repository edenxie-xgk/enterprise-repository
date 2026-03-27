from src.rag.rag_service import rag_service
from src.types.rag_type import RagContext


def rag_tool(query:RagContext,user_context:dict):
    return rag_service.query(query,user_context)