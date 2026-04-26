from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agent.profile_utils import build_preferred_topics_note, merge_queries_with_topic_guidance
from src.config.llm_config import LLMService
from src.prompts.agent.expand import EXPAND_PROMPT
from src.types.base_type import BaseNodeResult


class ExpandResult(BaseNodeResult):
    name: Optional[str] = Field(default="expand_query", description="tool name")
    answer: List[str] = Field(default_factory=list, description="expanded queries")


class ExpandStructuredResult(BaseModel):
    answer: List[str] = Field(default_factory=list, description="expanded queries")


def _normalize_expand_queries(queries: list[str], original_query: str, user_profile=None) -> list[str]:
    return merge_queries_with_topic_guidance(
        queries,
        original_query,
        user_profile,
        limit=3,
        max_topic_queries=1,
    )


def expand_query_tool(llm: BaseChatModel, query: str, chat_history=None, user_profile=None) -> ExpandResult:
    prompt = EXPAND_PROMPT.format(query=query, chat_history=chat_history or [])
    preferred_topics_note = build_preferred_topics_note(user_profile)
    if preferred_topics_note:
        prompt = f"{prompt}\n\n{preferred_topics_note}"

    try:
        payload = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=ExpandStructuredResult,
        )
        response = ExpandResult(
            success=True,
            name="expand_query",
            answer=_normalize_expand_queries(getattr(payload, "answer", []) or [], query, user_profile),
            message="expand query success",
            diagnostics=["expand_query_completed"],
        )
        if preferred_topics_note:
            response.diagnostics.append("preferred_topics_hint_applied:expand_query")
        return response
    except Exception as exc:
        return ExpandResult(
            success=False,
            answer=_normalize_expand_queries([], query, user_profile),
            name="expand_query",
            message="expand query failed",
            error_detail=str(exc),
            diagnostics=["expand_query_failed"],
        )
