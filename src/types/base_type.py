from typing import Any, Optional

from typing import Literal

from pydantic import BaseModel, Field


FailReason = Literal[
    "no_data",
    "low_recall",
    "bad_ranking",
    "ambiguous_query",
    "insufficient_context",
    "verification_failed",
    "tool_error",
    "permission_denied",
    "timeout",
    "max_steps_exceeded",
]


class BaseResult(BaseModel):
    success: bool = Field(default=False, description="是否成功")
    message: Optional[str] = Field(default="", description="消息")
    error_code: str | None = Field(default=None, description="错误码")
    error_detail: str | None = Field(default=None, description="错误详情")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    diagnostics: list[str] = Field(default_factory=list, description="诊断信息")


class BaseNodeResult(BaseResult):
    """给 reasoning node 用"""

    answer: Optional[Any] = Field(default=None, description="回答")


class BaseToolResult(BaseResult):
    tool_name: Optional[str] = Field(default=None, description="工具名称")
    answer: Optional[Any] = Field(default=None, description="回答")
    is_sufficient: bool = Field(default=False, description="是否足够回答")
    reason: Optional[str] = Field(default=None, description="诊断理由")
    fail_reason: Optional[FailReason] = Field(default=None, description="失败原因")


class BaseLLMDecideResult(BaseResult):
    """LLM 诊断返回的数据"""

    reason: Optional[str] = Field(default=None, description="诊断信息")
    confidence: Optional[float] = Field(default=None, description="置信度")
    next_action: Optional[str] = Field(default=None, description="下一步路由")
