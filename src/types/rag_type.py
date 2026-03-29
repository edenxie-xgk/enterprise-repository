from typing import TypedDict, List, Optional, Literal

from pydantic import BaseModel, Field

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.types.base_type import BaseToolResult


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


class DocumentInfo(BaseModel):
    content:str = Field(default="",description="文档内容")
    metadata:Optional[DocumentMetadata] = Field(default_factory=lambda:DocumentMetadata(),description="文档信息")
    dense_score:Optional[float] = Field(default=None,description="向量分数")
    bm25_score:Optional[float] = Field(default=None,description="语义相似度分数")
    rerank_score:Optional[float] = Field(default=None,description="重排分数")
    node_id:Optional[str] = Field(default="",description="片段ID")


class RAGResult(BaseToolResult):
    tool_name : Optional[str] =  'rag'
    # ===== 检索信息 =====
    documents: List[DocumentInfo] = Field(default=[], description="检索到的文档")

    # ===== 诊断信息（给Agent用）=====
    fail_reason:  Literal[
        "low_recall",      # 没召回
        "bad_ranking",     # 排序差
        "ambiguous_query", # query不清晰
        "no_data",         # 没数据
        "verification_failed", # 验证不通过
        "nothing_abnormal", #没有异常
    ] = Field(default=None,description="诊断信息")

    # # ===== 行为建议（关键设计）=====
    # suggested_actions: List[Literal[
    #     "retry",
    #     "rewrite",
    #     "expand",
    #     "decompose",
    #     "retrieval",
    #     "rerank",
    #     "abort"
    # ]] = Field(default=None,description="行为建议")


