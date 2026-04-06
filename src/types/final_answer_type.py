from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.types.base_type import FailReason


class FinalAnswerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = Field(default=True, description="是否成功生成最终答案")
    message: Optional[str] = Field(default="", description="最终处理消息")
    error_detail: Optional[str] = Field(default=None, description="最终处理错误详情")
    diagnostics: List[str] = Field(default_factory=list, description="最终处理诊断信息")
    answer: str = Field(default="", description="面向用户的最终答案")
    citations: List[str] = Field(default_factory=list, description="支持的引用来源")
    reason: Optional[str] = Field(default=None, description="形成此最终答案的原因")
    fail_reason: Optional[FailReason] = Field(default=None, description="当上下文不充分时的失败原因")
