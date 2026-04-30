from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config.llm_config import LLMService
from src.prompts.rag.evidence_prompt import EVIDENCE_PROMPT


class EvidenceAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_summary: str = Field(default="", description="基于证据的摘要")
    citations: list[str] = Field(default_factory=list, description="引用节点 ID")
    is_sufficient: bool = Field(default=False, description="证据是否充分")
    fail_reason: Literal[
        "low_recall",
        "bad_ranking",
        "ambiguous_query",
        "no_data",
        "insufficient_context",
    ] | None = Field(default=None, description="证据不足时的失败原因")


def evaluate_evidence(llm: BaseChatModel, query: str, context: str) -> EvidenceAssessmentResult:
    prompt = EVIDENCE_PROMPT.format(query=query, context=context)
    response: EvidenceAssessmentResult = LLMService.invoke(
        llm=llm,
        messages=[HumanMessage(content=prompt)],
        schema=EvidenceAssessmentResult,
    )
    if response is None:
        return EvidenceAssessmentResult(
            evidence_summary="",
            citations=[],
            is_sufficient=False,
            fail_reason="insufficient_context",
        )
    return response


# 重构期间保留的向后兼容别名。
generate_answer = evaluate_evidence
