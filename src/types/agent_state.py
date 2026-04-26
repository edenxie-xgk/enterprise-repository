from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.types.event_type import BaseEvent
from src.types.graph_type import GraphQueryContext
from src.types.memory_type import MemoryRecallResult, MemoryRecord
from src.types.rag_type import RAGResult, RagContext, SubQueryResult
from src.types.trace_type import TraceRecord


class State(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    query: Optional[str]
    run_id: Optional[str] = Field(default="", description="运行ID")
    user_id: Optional[str] = Field(default="", description="用户ID")
    session_id: Optional[str] = Field(default="", description="会话ID")
    output_level: Literal["concise", "standard", "detailed"] = Field(default="standard", description="输出详细程度")

    resolved_query: Optional[str] = Field(default="", description="解析后的查询")
    working_query: Optional[str] = Field(default="", description="当前工作查询")
    rewrite_query: Optional[str] = Field(default="", description="重写后的查询")
    expand_query: List[str] = Field(default_factory=list, description="扩展后的查询列表")
    decompose_query: List[str] = Field(default_factory=list, description="分解后的子查询列表")

    answer: Optional[str] = Field(default="", description="最终答案")
    citations: List[str] = Field(default_factory=list, description="最终引用来源")
    reason: Optional[str] = Field(default="", description="当前推理备注")
    action: Optional[str] = Field(default="", description="下一个动作")

    chat_history: List[str] = Field(default_factory=list, description="近期对话历史")
    user_profile: Optional[Dict[str, Any]] = Field(default=None, description="用户画像负载")
    short_term_memory: List[str] = Field(default_factory=list, description="短期记忆条目")
    working_memory: Optional[str] = Field(default=None, description="可直接用于提示词的工作记忆")
    long_term_memory_hits: List[MemoryRecord] = Field(default_factory=list, description="召回的长时记忆")
    long_term_memory_context: Optional[str] = Field(default=None, description="可直接用于提示词的长时记忆上下文")
    long_term_memory_used: bool = Field(default=False, description="是否使用了长时记忆")
    last_memory_result: Optional[MemoryRecallResult] = Field(default=None, description="最近一次长时记忆召回结果")
    memory_write_summary: Dict[str, Any] = Field(default_factory=dict, description="长时记忆回写摘要")

    last_rag_context: Optional[RagContext] = Field(default=None, description="最近的 RAG 上下文")
    last_rag_result: Optional[RAGResult] = Field(default=None, description="最近的 RAG 结果")
    last_graph_context: Optional[GraphQueryContext] = Field(default=None, description="最近的图谱检索上下文")
    last_graph_result: Optional[RAGResult] = Field(default=None, description="最近的图谱检索结果")
    sub_query_results: List[SubQueryResult] = Field(default_factory=list, description="子查询结果")

    action_history: List[BaseEvent] = Field(default_factory=list, description="动作历史")
    trace: List[TraceRecord] = Field(default_factory=list, description="追踪记录")
    diagnostics: List[str] = Field(default_factory=list, description="诊断信息")
    current_step: int = Field(default=0, description="当前步骤数")
    max_steps: int = Field(default=20, description="最大步骤数")
    duration_ms: int = Field(default=0, description="总耗时（毫秒）")
    llm_usage: Dict[str, Any] = Field(default_factory=dict, description="LLM 使用情况摘要")

    status: Literal["pending", "success", "failed"] = Field(default="pending", description="运行状态")
    fail_reason: Optional[str] = Field(default=None, description="失败原因")
    retry_count: int = Field(default=0, description="重试次数")
