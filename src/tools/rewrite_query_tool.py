from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agent.profile_utils import build_preferred_topics_note
from src.config.llm_config import LLMService
from src.prompts.agent.rewrite_prompt import REWRITE_PROMPT
from src.types.base_type import BaseNodeResult


class RewriteResult(BaseNodeResult):
    name: Optional[str] = Field(default="rewrite_query", description="tool name")
    max_attempt: Optional[int] = Field(default=2, description="max attempts")
    answer: Optional[str] = Field(default="", description="rewritten query")


class RewriteStructuredResult(BaseModel):
    answer: Optional[str] = Field(default="", description="rewritten query")


def rewrite_query_tool(
    llm: BaseChatModel,
    query: str,
    chat_history=None,
    user_profile=None,
    long_term_memory_context: str | None = None,
) -> RewriteResult:
    prompt = REWRITE_PROMPT.format(
        query=query,
        chat_history=chat_history or [],
    )
    preferred_topics_note = build_preferred_topics_note(user_profile)
    if preferred_topics_note:
        prompt = f"{prompt}\n\n{preferred_topics_note}"
    if long_term_memory_context and long_term_memory_context.strip():
        prompt = (
            f"{prompt}\n\n"
            "[长期记忆]\n"
            f"{long_term_memory_context.strip()}\n\n"
            "使用规则:\n"
            "- 只把这些长期记忆当作弱上下文。\n"
            "- 仅在它们能帮助补全指代、省略或稳定背景时使用。\n"
            "- 不要让长期记忆覆盖用户当前问题的真实意图。"
        )

    try:
        payload = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=RewriteStructuredResult,
        )
        answer = (getattr(payload, "answer", None) or query or "").strip()
        response = RewriteResult(
            success=True,
            name="rewrite_query",
            answer=answer,
            message="rewrite query success",
            diagnostics=["rewrite_query_completed"],
        )
        if preferred_topics_note:
            response.diagnostics.append("preferred_topics_hint_applied:rewrite_query")
        if long_term_memory_context and long_term_memory_context.strip():
            response.diagnostics.append("long_term_memory_hint_applied:rewrite_query")
        return response
    except Exception as exc:
        return RewriteResult(
            answer=(query or "").strip(),
            success=False,
            name="rewrite_query",
            message="rewrite query failed",
            error_detail=str(exc),
            diagnostics=["rewrite_query_failed"],
        )
