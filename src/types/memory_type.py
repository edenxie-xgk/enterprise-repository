from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.types.base_type import BaseResult


MemoryBackend = Literal["disabled", "milvus"]
MemoryScope = Literal["user", "session"]
MemoryType = Literal["preference", "identity_fact", "task_context", "constraint"]
MemorySource = Literal["user_explicit", "assistant_extract", "profile_sync"]


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(default="", description="记忆唯一标识")
    user_id: str = Field(default="", description="用户ID")
    session_id: str | None = Field(default=None, description="会话ID")
    scope: MemoryScope = Field(default="user", description="记忆可见范围")
    memory_type: MemoryType = Field(default="task_context", description="记忆语义类型")
    content: str = Field(default="", description="记忆原始内容")
    summary: str = Field(default="", description="记忆简短摘要")
    tags: list[str] = Field(default_factory=list, description="记忆标签")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="记忆重要性")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="记忆置信度")
    source: MemorySource = Field(default="assistant_extract", description="记忆来源")
    dedupe_key: str = Field(default="", description="用于 upsert 和去重的稳定键")
    created_at: str = Field(default="", description="创建时间")
    updated_at: str = Field(default="", description="最后更新时间")
    last_accessed_at: str | None = Field(default=None, description="最后召回时间")
    expires_at: str | None = Field(default=None, description="可选过期时间")
    is_active: bool = Field(default=True, description="记忆是否活跃")
    score: float | None = Field(default=None, description="召回分数")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class MemoryRecallQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="", description="用户ID")
    session_id: str | None = Field(default=None, description="当前会话ID")
    query: str = Field(default="", description="用于召回的查询")
    top_k: int = Field(default=3, ge=1, le=20, description="返回的最大记忆数量")
    min_score: float = Field(default=0.35, ge=0.0, le=1.0, description="最低相似度分数")
    scopes: list[MemoryScope] = Field(default_factory=lambda: ["user"], description="允许的范围")
    memory_types: list[MemoryType] = Field(default_factory=list, description="可选的记忆类型过滤器")


class MemoryRecallResult(BaseResult):
    memories: list[MemoryRecord] = Field(default_factory=list, description="召回的記憶")
    memory_context: str = Field(default="", description="可直接用于提示词的记忆上下文")
    used: bool = Field(default=False, description="是否使用了记忆")


class MemoryWriteCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType = Field(default="task_context", description="候选记忆类型")
    scope: MemoryScope = Field(default="user", description="候选范围")
    content: str = Field(default="", description="记忆内容")
    summary: str = Field(default="", description="记忆摘要")
    tags: list[str] = Field(default_factory=list, description="记忆标签")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性分数")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="置信度分数")
    source: MemorySource = Field(default="assistant_extract", description="候选来源")
    dedupe_key: str = Field(default="", description="用于去重的稳定键")
    expires_at: Optional[str] = Field(default=None, description="可选过期时间")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class MemoryWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="", description="用户ID")
    session_id: str | None = Field(default=None, description="会话ID")
    query: str = Field(default="", description="原始查询")
    answer: str = Field(default="", description="最终答案")
    chat_history: list[str] = Field(default_factory=list, description="近期对话历史")
    user_profile: dict[str, Any] = Field(default_factory=dict, description="用户画像快照")
    existing_memories: list[MemoryRecord] = Field(default_factory=list, description="已召回的記憶")


class MemoryWriteResult(BaseResult):
    written_count: int = Field(default=0, description="写入的记忆数量")
    skipped_count: int = Field(default=0, description="跳过的记忆数量")
    memory_ids: list[str] = Field(default_factory=list, description="写入或更新的记忆ID")
    candidates: list[MemoryWriteCandidate] = Field(default_factory=list, description="生成的候选记忆")