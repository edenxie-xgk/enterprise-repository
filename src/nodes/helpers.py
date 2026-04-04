import time
from uuid import uuid4
from typing import Any, Optional, TypeVar

from src.memory import build_memory_entry, build_working_memory, compact_short_term_memory
from src.types.agent_state import State
from src.types.base_type import BaseResult
from src.types.event_type import BaseEvent
from src.types.trace_type import TraceRecord
from utils.utils import get_current_time


TEvent = TypeVar("TEvent", bound=BaseEvent)


def create_event(event_cls, name: str, input_data=None, max_attempt: int = 3):
    return event_cls(
        name=name,
        input=input_data,
        max_attempt=max_attempt,
        started_at=get_current_time(),
    )


def get_next_attempt(action_history: list[BaseEvent], event_name: str) -> int:
    last_event = next((event for event in reversed(action_history) if event.name == event_name), None)
    if last_event:
        return last_event.attempt + 1
    return 1


def finalize_event(event: TEvent, result: Optional[BaseResult], start_time: float) -> TEvent:
    event.output = result
    event.status = "success" if result and result.success else "failed"
    event.error = result.error_detail if result else None
    event.ended_at = get_current_time()
    event.duration = int((time.time() - start_time) * 1000)
    return event


def build_state_patch(state: State, event: BaseEvent, **updates: Any) -> dict[str, Any]:
    """构建状态更新补丁

    根据当前状态和事件信息，生成状态更新的字典补丁

    Args:
        state: 当前状态对象
        event: 需要处理的基础事件
        **updates: 其他需要合并到补丁中的键值对

    Returns:
        返回包含状态更新信息的字典
    """
    # 复制当前诊断信息并追加新记录
    diagnostics = list(state.diagnostics)
    diagnostics.append(f"{event.name}:{event.status}:{event.duration or 0}ms")

    # 计算下一步骤编号
    next_step = state.current_step + 1

    # 复制当前跟踪记录并添加新事件记录
    trace = list(state.trace)
    trace.append(
        TraceRecord(
            step=next_step,  # 步骤编号
            event_id=event.id,  # 事件ID
            event_kind=event.kind,  # 事件类型
            event_name=event.name,  # 事件名称
            status=event.status,  # 事件状态(success/failed)
            attempt=event.attempt,  # 尝试次数
            duration_ms=event.duration or 0,  # 持续时间(毫秒)
            started_at=event.started_at,  # 开始时间
            ended_at=event.ended_at,  # 结束时间
            fail_reason=getattr(event.output, "fail_reason", None) if event.output else None,  # 失败原因
            message=getattr(event.output, "message", None) if event.output else None,  # 输出消息
            diagnostics=list(getattr(event.output, "diagnostics", []) or []),  # 输出诊断信息
        )
    )

    # 构建基础补丁字典
    diagnostics.append(f"trace_step={next_step}")
    short_term_memory = compact_short_term_memory(state.short_term_memory + [build_memory_entry(event)])
    working_memory = build_working_memory(short_term_memory)

    patch = {
        "action_history": state.action_history + [event],  # 更新操作历史
        "trace": trace,  # 更新跟踪记录
        "run_id": state.run_id or str(uuid4()),  # 设置或生成运行ID
        "current_step": next_step,  # 更新当前步骤
        "short_term_memory": short_term_memory,
        "working_memory": working_memory,
        "diagnostics": diagnostics,  # 更新诊断信息
    }

    # 合并额外更新字段
    patch.update(updates)
    return patch
