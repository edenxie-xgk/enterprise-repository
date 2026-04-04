from typing import Literal, Optional

from pydantic import BaseModel, Field


TraceStatus = Literal["pending", "success", "failed"]


class TraceRecord(BaseModel):
    step: int = Field(default=0, description="运行步数")
    event_id: str = Field(default="", description="事件ID")
    event_kind: str = Field(default="", description="事件类型")
    event_name: str = Field(default="", description="事件名称")
    status: TraceStatus = Field(default="pending", description="执行状态")
    attempt: int = Field(default=0, description="尝试次数")
    duration_ms: int = Field(default=0, description="耗时")
    started_at: Optional[str] = Field(default=None, description="开始时间")
    ended_at: Optional[str] = Field(default=None, description="结束时间")
    fail_reason: Optional[str] = Field(default=None, description="失败原因")
    message: Optional[str] = Field(default=None, description="消息")
    diagnostics: list[str] = Field(default_factory=list, description="诊断信息")
