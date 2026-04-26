from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.types.base_type import BaseResult, BaseToolResult, FailReason


GraphEntityType = Literal["company", "report", "section", "metric", "topic", "counterparty"]
GraphFactKind = Literal["metric", "event", "risk", "related_party", "policy", "management_view"]
GraphQueryKind = Literal[
    "metric_lookup",
    "period_comparison",
    "event_lookup",
    "risk_lookup",
    "related_party_lookup",
    "general",
]


class GraphEvidence(BaseModel):
    node_id: str = Field(default="", description="原始 chunk 节点 ID")
    content: str = Field(default="", description="证据文本快照")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="证据元数据")


class GraphEntity(BaseModel):
    entity_id: str = Field(default="", description="实体唯一 ID")
    entity_type: GraphEntityType = Field(default="topic", description="实体类型")
    name: str = Field(default="", description="实体名")
    normalized_name: str = Field(default="", description="标准化实体名")
    aliases: list[str] = Field(default_factory=list, description="实体别名")
    department_id: int | None = Field(default=None, description="所属部门 ID")
    report_id: str | None = Field(default=None, description="所属报告 ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加属性")
    search_terms: list[str] = Field(default_factory=list, description="检索关键字")


class FinancialFact(BaseModel):
    fact_id: str = Field(default="", description="事实唯一 ID")
    fact_key: str = Field(default="", description="用于 upsert 的稳定键")
    fact_kind: GraphFactKind = Field(default="metric", description="事实类型")
    company_name: str = Field(default="", description="公司名")
    normalized_company_name: str = Field(default="", description="标准化公司名")
    report_id: str = Field(default="", description="报告 ID")
    report_name: str = Field(default="", description="报告名")
    section_title: str = Field(default="", description="片段所属章节标题")
    topic: str = Field(default="", description="主题")
    metric_name: str | None = Field(default=None, description="指标名")
    normalized_metric_name: str | None = Field(default=None, description="标准化指标名")
    summary: str = Field(default="", description="事实摘要")
    raw_value: str | None = Field(default=None, description="原始值")
    numeric_value: float | None = Field(default=None, description="标准化数值")
    unit: str | None = Field(default=None, description="数值单位")
    currency: str | None = Field(default=None, description="币种")
    period_end: str | None = Field(default=None, description="报告期结束日 ISO 字符串")
    period_year: str | None = Field(default=None, description="报告年份")
    period_type: str | None = Field(default=None, description="报告类型")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="抽取置信度")
    department_id: int | None = Field(default=None, description="所属部门 ID")
    evidence_node_ids: list[str] = Field(default_factory=list, description="证据 chunk 节点 ID 列表")
    evidence_docs: list[GraphEvidence] = Field(default_factory=list, description="证据文本快照")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加属性")
    search_terms: list[str] = Field(default_factory=list, description="检索关键字")
    search_text: str = Field(default="", description="归一化检索文本")


class GraphExtractionBundle(BaseModel):
    entities: list[GraphEntity] = Field(default_factory=list, description="抽取到的实体列表")
    facts: list[FinancialFact] = Field(default_factory=list, description="抽取到的事实列表")
    diagnostics: list[str] = Field(default_factory=list, description="抽取诊断信息")


class GraphQueryInterpretation(BaseModel):
    query_kind: GraphQueryKind = Field(default="general", description="图检索问题类型")
    metric_names: list[str] = Field(default_factory=list, description="识别到的标准化指标名")
    topics: list[str] = Field(default_factory=list, description="识别到的主题列表")
    years: list[str] = Field(default_factory=list, description="识别到的年份列表")
    company_terms: list[str] = Field(default_factory=list, description="识别到的公司词")
    search_terms: list[str] = Field(default_factory=list, description="最终检索词")
    comparison_mode: bool = Field(default=False, description="是否为比较/趋势类问题")
    diagnostics: list[str] = Field(default_factory=list, description="诊断信息")


class GraphQueryContext(BaseModel):
    query: Optional[str] = Field(default="", description="原始查询")
    rewritten_query: Optional[str] = Field(default="", description="改写查询")
    expand_query: list[str] = Field(default_factory=list, description="扩展查询列表")
    decompose_query: list[str] = Field(default_factory=list, description="拆解查询列表")
    filters: dict[str, Any] = Field(default_factory=dict, description="访问控制过滤器")
    top_k: int = Field(default=settings.graph_query_top_k, description="返回事实数量")
    max_candidate_facts: int = Field(default=settings.graph_query_max_candidates, description="候选事实上限")
    query_kind: GraphQueryKind = Field(default="general", description="图查询问题类型")
    metric_names: list[str] = Field(default_factory=list, description="标准化指标名")
    topics: list[str] = Field(default_factory=list, description="主题")
    years: list[str] = Field(default_factory=list, description="年份")
    company_terms: list[str] = Field(default_factory=list, description="公司词")
    search_terms: list[str] = Field(default_factory=list, description="最终检索词")
    comparison_mode: bool = Field(default=False, description="比较模式")


class GraphBuildResult(BaseResult):
    entity_count: int = Field(default=0, description="写入实体数量")
    fact_count: int = Field(default=0, description="写入事实数量")


class GraphSearchResult(BaseToolResult):
    name: str = "graph_rag"
    facts: list[FinancialFact] = Field(default_factory=list, description="命中的图谱事实")
    citations: list[str] = Field(default_factory=list, description="证据引用 node_id")
    retrieval_queries: list[str] = Field(default_factory=list, description="检索查询列表")
    retrieval_candidate_node_ids: list[str] = Field(default_factory=list, description="候选事实 ID")
    rerank_node_ids: list[str] = Field(default_factory=list, description="保留事实 ID")
    fail_reason: Optional[FailReason] = Field(default=None, description="失败原因")
