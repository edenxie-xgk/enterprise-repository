from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.types.base_type import BaseToolResult, FailReason


class RagContext(BaseModel):
    query: Optional[str] = Field(default="", description="原始查询")
    rewritten_query: Optional[str] = Field(default="", description="重写查询")
    expand_query: List[str] = Field(default_factory=list, description="扩展查询")
    decompose_query: List[str] = Field(default_factory=list, description="子任务规划")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤器")
    retrieval_top_k: int = Field(default=settings.retriever_top_k, description="向量召回数量")
    rerank_top_k: int = Field(default=settings.reranker_top_k, description="重排数量")
    use_retrieval: bool = Field(default=True, description="是否需要重新召回")
    use_rerank: bool = Field(default=True, description="是否需要重新重排")


class DocumentInfo(BaseModel):
    node_id: Optional[str] = Field(default="", description="片段ID")
    content: str = Field(default="", description="文档内容")
    metadata: DocumentMetadata = Field(default_factory=lambda: DocumentMetadata(), description="文档信息")
    dense_score: Optional[float] = Field(default=None, description="向量分数")
    bm25_score: Optional[float] = Field(default=None, description="关键词分数")
    rerank_score: Optional[float] = Field(default=None, description="重排分数")


class SubQueryResult(BaseModel):
    sub_query: str = Field(default="", description="子问题")
    answer: str = Field(default="", description="子问题回答")
    citations: List[str] = Field(default_factory=list, description="子问题引用")
    documents: List[DocumentInfo] = Field(default_factory=list, description="子问题文档")
    is_sufficient: bool = Field(default=False, description="是否足够回答子问题")
    fail_reason: Optional[FailReason] = Field(default=None, description="子问题失败原因")
    diagnostics: List[str] = Field(default_factory=list, description="子问题诊断信息")


class RAGResult(BaseToolResult):
    documents: List[DocumentInfo] = Field(default_factory=list, description="检索到的文档")
    citations: List[str] = Field(default_factory=list, description="生成答案中的引用")
    is_sufficient: bool = Field(default=False, description="是否足够回答")
    fail_reason: Optional[FailReason] = Field(default=None, description="诊断信息")
    confidence: Optional[float] = Field(default=None, description="置信度")
    retrieval_queries: List[str] = Field(default_factory=list, description="检索查询")
    diagnostics: List[str] = Field(default_factory=list, description="诊断信息")
