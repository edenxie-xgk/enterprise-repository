from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.types.base_type import BaseToolResult, FailReason


class RagContext(BaseModel):
    query: Optional[str] = Field(default="", description="原始查询")
    rewritten_query: Optional[str] = Field(default="", description="重写后的查询")
    expand_query: List[str] = Field(default_factory=list, description="扩展后的查询列表")
    decompose_query: List[str] = Field(default_factory=list, description="分解后的子查询列表")
    filters: Dict[str, Any] = Field(default_factory=dict, description="元数据过滤器")
    retrieval_top_k: int = Field(default=settings.retriever_top_k, description="检索返回的 top-k 数量")
    rerank_top_k: int = Field(default=settings.reranker_top_k, description="重排序返回的 top-k 数量")
    use_retrieval: bool = Field(default=True, description="是否重新执行检索")
    use_rerank: bool = Field(default=True, description="是否重新执行重排序")


class DocumentInfo(BaseModel):
    node_id: Optional[str] = Field(default="", description="分块节点 ID")
    content: str = Field(default="", description="分块内容")
    metadata: DocumentMetadata = Field(default_factory=lambda: DocumentMetadata(), description="文档元数据")
    dense_score: Optional[float] = Field(default=None, description="稠密检索分数")
    bm25_score: Optional[float] = Field(default=None, description="BM25 分数")
    rerank_score: Optional[float] = Field(default=None, description="重排序分数")


class SubQueryResult(BaseModel):
    sub_query: str = Field(default="", description="子查询")
    answer: str = Field(default="", description="向后兼容的答案字段")
    evidence_summary: str = Field(default="", description="子查询的证据摘要")
    citations: List[str] = Field(default_factory=list, description="子查询的证据引用来源")
    retrieval_candidate_node_ids: List[str] = Field(default_factory=list, description="检索返回的候选节点 ID 列表")
    rerank_node_ids: List[str] = Field(default_factory=list, description="重排序后保留的节点 ID 列表")
    documents: List[DocumentInfo] = Field(default_factory=list, description="子查询检索到的文档")
    is_sufficient: bool = Field(default=False, description="证据是否充分")
    fail_reason: Optional[FailReason] = Field(default=None, description="失败原因")
    diagnostics: List[str] = Field(default_factory=list, description="诊断信息")


class RAGResult(BaseToolResult):
    documents: List[DocumentInfo] = Field(default_factory=list, description="检索到的文档")
    citations: List[str] = Field(default_factory=list, description="证据引用来源")
    evidence_summary: Optional[str] = Field(default="", description="证据摘要")
    is_sufficient: bool = Field(default=False, description="证据是否充分")
    fail_reason: Optional[FailReason] = Field(default=None, description="失败原因")
    confidence: Optional[float] = Field(default=None, description="置信度")
    retrieval_queries: List[str] = Field(default_factory=list, description="用于检索的查询列表")
    retrieval_candidate_node_ids: List[str] = Field(default_factory=list, description="重排序前返回的节点 ID 列表")
    rerank_node_ids: List[str] = Field(default_factory=list, description="重排序后保留的节点 ID 列表")
    diagnostics: List[str] = Field(default_factory=list, description="诊断信息")
