from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.congfig.llm_config import LLMService
from src.prompts.rag.evidence_prompt import EVIDENCE_PROMPT


class EvidenceAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_summary: str = Field(default="", description="Evidence-oriented summary")
    citations: list[str] = Field(default_factory=list, description="Citation node ids")
    is_sufficient: bool = Field(default=False, description="Whether evidence is sufficient")
    fail_reason: Literal[
        "low_recall",
        "bad_ranking",
        "ambiguous_query",
        "no_data",
        "insufficient_context",
    ] | None = Field(default=None, description="Failure reason when evidence is insufficient")


def evaluate_evidence(llm: BaseChatModel, query: str, context: str) -> EvidenceAssessmentResult:
    prompt = EVIDENCE_PROMPT.format(query=query, context=context)
    response: EvidenceAssessmentResult = LLMService.invoke(
        llm=llm,
        messages=[HumanMessage(content=prompt)],
        schema=EvidenceAssessmentResult,
    )
    return response


# Backward-compatible alias during refactor.
generate_answer = evaluate_evidence
