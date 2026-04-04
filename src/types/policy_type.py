from typing import Optional

from pydantic import BaseModel, Field


class InputGuardDecision(BaseModel):
    is_valid: bool = Field(default=True, description="输入是否有效")
    reason: Optional[str] = Field(default=None, description="拦截原因")
    response: Optional[str] = Field(default=None, description="面向用户的回复")


class InitialActionDecision(BaseModel):
    next_action: str = Field(default="rag", description="初始动作")
    reason: Optional[str] = Field(default=None, description="决策理由")
    clarification_question: Optional[str] = Field(default=None, description="澄清问题")


class RetrievalPolicyPlan(BaseModel):
    retrieval_top_k: int = Field(default=0, description="检索返回的 top-k 数量")
    rerank_top_k: int = Field(default=0, description="重排序返回的 top-k 数量")
    use_retrieval: bool = Field(default=True, description="是否重新执行检索")
    use_rerank: bool = Field(default=True, description="是否重新执行重排序")
    needs_more_recall: bool = Field(default=False, description="是否需要提高召回率")
    needs_more_precision: bool = Field(default=False, description="是否需要提高精确率")
    strategy_reason: Optional[str] = Field(default=None, description="策略理由")
