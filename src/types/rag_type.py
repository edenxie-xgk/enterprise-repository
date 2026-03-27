from typing import TypedDict, List, Optional, Literal

from pydantic import BaseModel, Field

from core.settings import settings


class RagContext(BaseModel):
    query: Optional[str] = Field(default="",description="原始查询")
    rewritten_query: Optional[str] = Field(default="", description="重写查询")

    expand_query: List[str] = Field(default=[], description="拓展查询")

    decompose_query: List[str] = Field(default=[], description="子任务规划")

    # ===== 检索控制 =====
    retrieval_top_k: int = Field(default=settings.retriever_top_k,description="向量召回数量")
    rerank_top_k: int = Field(default=settings.reranker_top_k,description="重排数量")
    use_retrieval:bool = Field(default=True,description="是否需要重新召回")
    use_rerank: bool = Field(default=True,description="是否需要重新重排")

    # ===== 历史信息（用于二次决策）=====
    previous_attempts: int = Field(default = 0,description="第几次尝试rag")       # 第几次尝试
    previous_fail_reason: Optional[str] = Field(default=[],description="rag历史信息")



class RAGResult(BaseModel):
    # ===== 最终输出 =====
    answer: Optional[str] = Field(default="",description="回答")

    # ===== 检索信息 =====
    documents: List[dict] = Field(default=[],description="检索到的文档")

    # ===== 质量评估（核心）=====
    confidence: float = Field(default=0.0,description="0~1（模型或规则估计）")
    coverage: float = Field(default=0.0,description="覆盖度（是否回答完整）")

    # ===== 诊断信息（给Agent用）=====
    is_sufficient: bool = Field(default=False,description="是否足够回答")
    fail_reason:  Literal[
        "low_recall",      # 没召回
        "bad_ranking",     # 排序差
        "ambiguous_query", # query不清晰
        "no_data",         # 没数据
    ] = Field(default=None,description="召回说明")

    # ===== 行为建议（关键设计）=====
    suggested_actions: List[Literal[
        "retry",
        "rewrite",
        "expand",
        "decompose",
        "abort"
    ]] = Field(default=None,description="行为建议")

    # ===== 调试信息 =====
    debug_info: Optional[dict] = Field(default=None,description="调试信息")