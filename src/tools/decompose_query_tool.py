from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field

from src.congfig.llm_config import LLMService
from src.prompts.agent.decompose import DECOMPOSE_PROMPT
from src.types.base_type import BaseNodeResult


class DecomposeResult(BaseNodeResult):
    tool_name: Optional[str] = Field(default="decompose_query", description="工具名称")
    answer: List[str] = Field(default_factory=list, description="拆解后的子问题")


def _normalize_sub_queries(queries: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for item in queries or []:
        candidate = (item or "").strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(candidate)
    return normalized[:3]


def decompose_query_tool(llm: BaseChatModel, query: str, chat_history=None) -> DecomposeResult:
    prompt = DECOMPOSE_PROMPT.format(query=query, chat_history=chat_history or [])
    try:
        response: DecomposeResult = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=DecomposeResult,
        )
        response.success = True
        response.tool_name = "decompose_query"
        response.answer = _normalize_sub_queries(response.answer)
        response.message = "decompose query success"
        response.diagnostics = list(response.diagnostics or []) + ["decompose_query_completed"]
        return response
    except Exception as exc:
        return DecomposeResult(
            answer=[],
            success=False,
            tool_name="decompose_query",
            message="decompose query failed",
            error_detail=str(exc),
            diagnostics=["decompose_query_failed"],
        )
