from __future__ import annotations

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.config.llm_config import LLMService
from src.prompts.rag.answer_verify import VERIFY_PROMPT


class AnswerVerifyResult(BaseModel):
    valid: bool = Field(..., description="Whether the answer is fully supported by the context")
    reason: Optional[str] = Field(default=None, description="Why the answer failed or passed")


def verify_answer(llm: BaseChatModel, context: str, answer: str) -> AnswerVerifyResult:
    prompt = VERIFY_PROMPT.format(
        context=context,
        answer=answer,
    )

    return LLMService.invoke(
        llm=llm,
        messages=[HumanMessage(content=prompt)],
        schema=AnswerVerifyResult,
    )
