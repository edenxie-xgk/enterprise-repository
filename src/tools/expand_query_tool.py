from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.congfig.llm_config import LLMService
from src.prompts.agent.expand import EXPAND_PROMPT

class ExpandResult(BaseModel):
    """
       查询扩展助手返回的数据
    """
    queries:List[str] = Field(...,description="扩展查询列表")

def expand_query(llm:BaseChatModel, query: str, chat_history=None):

    prompt = EXPAND_PROMPT.format(query=query, chat_history=chat_history or [])

    try:
        response: ExpandResult = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=ExpandResult
        )
        return response.queries
    except Exception:
        return []
