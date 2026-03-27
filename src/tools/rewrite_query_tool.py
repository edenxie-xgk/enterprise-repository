from typing import List, Dict, Literal

from anthropic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field
from src.congfig.llm_config import LLMService
from src.prompts.agent.rewrite_prompt import REWRITE_PROMPT



class RewriteResult(BaseModel):
    """
        Query优化助手返回的数据
    """
    rewrite_query: str = Field(...,description="原始语言")




def rewrite_query_tool(llm:BaseChatModel,query:str,chat_history=None,user_profile=None):
    """对输入的数据进行一个同义词替换、易检索更改"""
    prompt = REWRITE_PROMPT.format(
        query=query,
        chat_history=chat_history or []
    )
    try:
        response: RewriteResult = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=RewriteResult
        )
        return response.rewrite_query
    except Exception:
        return query

