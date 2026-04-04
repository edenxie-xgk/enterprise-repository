from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.types.event_type import BaseEvent
from src.types.rag_type import RAGResult, RagContext
from src.types.trace_type import TraceRecord


class State(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    query: Optional[str]
    run_id: Optional[str] = Field(default="", description="一次agent运行ID")
    user_id: Optional[str] = Field(default="", description="用户ID")
    session_id: Optional[str] = Field(default="", description="会话ID")

    normalized_query: Optional[str] = Field(default="", description="规范化查询")
    working_query: Optional[str] = Field(default="", description="当前工作查询")
    rewrite_query: Optional[str] = Field(default="", description="重写查询")
    expand_query: List[str] = Field(default_factory=list, description="扩展查询")
    decompose_query: List[str] = Field(default_factory=list, description="子任务规划")

    answer: Optional[str] = Field(default="", description="回答")
    reason: Optional[str] = Field(default="", description="推断结果")
    action: Optional[str] = Field(default="", description="下一步行动")

    chat_history: List[str] = Field(default_factory=list, description="短期记忆")
    user_profile: Optional[Dict[str, Any]] = Field(default=None, description="用户画像")
    short_term_memory: List[str] = Field(default_factory=list, description="短期记忆")
    working_memory: Optional[str] = Field(default=None, description="工作记忆")

    last_rag_context: Optional[RagContext] = Field(default=None, description="上一次RAG上下文")
    last_rag_result: Optional[RAGResult] = Field(default=None, description="上一次RAG结果")

    action_history: List[BaseEvent] = Field(default_factory=list, description="行动调用历史")
    trace: List[TraceRecord] = Field(default_factory=list, description="结构化运行轨迹")
    diagnostics: List[str] = Field(default_factory=list, description="运行诊断信息")
    current_step: int = Field(default=0, description="当前已执行步数")
    max_steps: int = Field(default=6, description="最大允许执行步数")

    status: Literal["pending", "success", "failed"] = Field(default="pending", description="状态")
    fail_reason: Optional[str] = Field(default=None, description="失败原因")
    retry_count: int = Field(default=0, description="重试次数")
