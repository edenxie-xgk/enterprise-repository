from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field

from src.congfig.llm_config import LLMService
from src.prompts.agent.expand import EXPAND_PROMPT
from src.types.base_type import BaseNodeResult


class ExpandResult(BaseNodeResult):
    tool_name: Optional[str] = Field(default="expand_query", description="工具名称")
    answer: List[str] = Field(default_factory=list, description="扩展查询")


def expand_query_tool(llm: BaseChatModel, query: str, chat_history=None) -> ExpandResult:
    prompt = EXPAND_PROMPT.format(query=query, chat_history=chat_history or [])

    try:
        response: ExpandResult = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=ExpandResult,
        )
        response.success = True
        response.tool_name = "expand_query"
        response.message = "expand query success"
        return response
    except Exception as exc:
        return ExpandResult(
            success=False,
            answer=[],
            tool_name="expand_query",
            message="expand query failed",
            error_detail=str(exc),
            diagnostics=["expand query failed"],
        )
