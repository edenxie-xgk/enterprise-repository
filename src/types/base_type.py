from typing import Any, Optional

from typing import Literal

from pydantic import BaseModel, Field


# 定义系统可能的失败原因枚举类型
# 每个值对应特定的错误场景，用于系统错误分类和处理
FailReason = Literal[
    "invalid_input",          # 输入无效或不合法
    "disallowed_query",       # 查询被禁止(如安全策略)
    "no_data",                # 无数据返回
    "low_recall",             # 召回率低(搜索结果不足)
    "bad_ranking",            # 结果排序质量差
    "ambiguous_query",        # 查询模糊不清
    "insufficient_context",   # 上下文信息不足
    "verification_failed",    # 验证失败
    "tool_error",             # 工具执行错误
    "permission_denied",      # 权限不足
    "timeout",                # 操作超时
    "max_steps_exceeded",     # 超过最大步骤限制
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
