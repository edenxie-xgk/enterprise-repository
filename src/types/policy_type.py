from typing import Optional

from pydantic import BaseModel, Field

from src.types.base_type import BaseLLMDecideResult


class InputGuardDecision(BaseModel):
    is_valid: bool = Field(default=True, description="whether the input can proceed")
    reason: Optional[str] = Field(default=None, description="why the input was blocked")
    response: Optional[str] = Field(default=None, description="user-facing block message")


class InitialActionDecision(BaseModel):
    next_action: str = Field(default="rag", description="initial action suggested by policy")
    reason: Optional[str] = Field(default=None, description="policy reason")
    clarification_question: Optional[str] = Field(default=None, description="clarifying question when needed")


class RetrievalPolicyPlan(BaseModel):
    retrieval_top_k: int = Field(default=0, description="retriever top-k")
    rerank_top_k: int = Field(default=0, description="reranker top-k")
    use_retrieval: bool = Field(default=True, description="whether retrieval should run")
    use_rerank: bool = Field(default=True, description="whether reranking should run")
    needs_more_recall: bool = Field(default=False, description="whether recall should be increased")
    needs_more_precision: bool = Field(default=False, description="whether precision should be increased")
    strategy_reason: Optional[str] = Field(default=None, description="reason for the retrieval strategy")


class AgentPlannerStructuredDecision(BaseModel):
    next_action: str = Field(default="finish", description="planner selected action")
    reason: Optional[str] = Field(default=None, description="why this action was selected")
    confidence: Optional[float] = Field(default=None, description="planner confidence score")
    clarification_question: Optional[str] = Field(default=None, description="clarifying question when needed")


class AgentPlannerDecision(BaseLLMDecideResult):
    next_action: str = Field(default="finish", description="planner selected action")
    clarification_question: Optional[str] = Field(default=None, description="clarifying question when needed")
