from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field

from src.congfig.llm_config import LLMService
from src.prompts.agent.rewrite_prompt import REWRITE_PROMPT
from src.types.base_type import BaseNodeResult


class RewriteResult(BaseNodeResult):
    tool_name: Optional[str] = Field(default="rewrite_query", description="工具名称")
    max_attempt: Optional[int] = Field(default=2, description="最大调用次数")
    answer: Optional[str] = Field(default="", description="改写后的查询")

def rewrite_query_tool(
    llm: BaseChatModel,
    query: str,
    chat_history=None,
    user_profile=None,
) -> RewriteResult:
    prompt = REWRITE_PROMPT.format(
        query=query,
        chat_history=chat_history or [],
    )

    try:
        response: RewriteResult = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=RewriteResult,
        )
        response.success = True
        response.tool_name = "rewrite_query"
        response.answer = (response.answer or query or "").strip()
        response.message = "rewrite query success"
        response.diagnostics = list(response.diagnostics or []) + ["rewrite_query_completed"]
        return response
    except Exception as exc:
        return RewriteResult(
            answer=(query or "").strip(),
            success=False,
            tool_name="rewrite_query",
            message="rewrite query failed",
            error_detail=str(exc),
            diagnostics=["rewrite_query_failed"],
        )
