from __future__ import annotations

from typing import Any

from src.types.event_type import BaseEvent


MAX_SHORT_TERM_MEMORY = 8


def _stringify(value: Any, max_len: int = 120) -> str:
    """将任意值转换为格式化后的字符串

    处理None值、字符串转换、换行符替换和长度截断，用于生成简洁的文本表示

    Args:
        value: 需要转换的任意值
        max_len: 最大允许长度，默认120个字符

    Returns:
        str: 格式化后的字符串，可能被截断并添加省略号
    """
    if value is None:
        return ""
    # 将非字符串值转换为字符串
    text = value if isinstance(value, str) else str(value)
    # 替换换行符为空格并去除首尾空白
    text = text.replace("\n", " ").strip()
    # 如果超过最大长度，截断并添加省略号
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def build_memory_entry(event: BaseEvent) -> str:
    """构建事件的内存条目字符串表示

    从BaseEvent对象中提取关键信息，构建格式化的字符串条目，用于工作记忆存储。
    包含事件类型、名称、状态、尝试次数，以及可选的失败原因和回答内容。

    Args:
        event: 基础事件对象，包含需要记录的事件信息

    Returns:
        str: 格式化后的字符串，各字段用竖线分隔。例如：
            "event:query | status=success | attempt=1 | answer=42"
    """
    # 安全获取event.output中的fail_reason和answer属性，避免AttributeError
    fail_reason = getattr(event.output, "fail_reason", None) if event.output else None
    answer = getattr(event.output, "answer", None) if event.output else None
    # 对answer进行字符串格式化处理，限制最大长度
    answer_text = _stringify(answer, max_len=80)

    # 构建基础信息段
    segments = [
        f"{event.kind}:{event.name}",  # 事件类型:名称
        f"status={event.status}",      # 事件状态
        f"attempt={event.attempt}",    # 尝试次数
    ]

    # 如果有失败原因，添加到信息段
    if fail_reason:
        segments.append(f"fail_reason={fail_reason}")
    # 如果有回答内容，添加到信息段
    if answer_text:
        segments.append(f"answer={answer_text}")

    # 用竖线连接所有信息段作为最终结果
    return " | ".join(segments)


def compact_short_term_memory(memory_items: list[str], max_items: int = MAX_SHORT_TERM_MEMORY) -> list[str]:
    """压缩短期记忆列表，保留最近的max_items个有效条目

    对输入的字符串列表进行规范化处理，去除空项和空白字符，
    并确保返回的条目数量不超过max_items限制。
    当条目数超过限制时，保留最新的条目（列表末尾的项）。

    Args:
        memory_items: 原始记忆条目列表，可能包含空字符串或空白字符
        max_items: 最大保留条目数，默认为MAX_SHORT_TERM_MEMORY(8)

    Returns:
        list[str]: 规范化且长度不超过max_items的字符串列表
    """
    # 规范化处理：去除空项和所有条目的首尾空白
    normalized = [item.strip() for item in memory_items if item and item.strip()]

    # 如果规范化后的条目数不超过限制，直接返回
    if len(normalized) <= max_items:
        return normalized

    # 超过限制时，只保留最新的max_items个条目（列表末尾的项）
    return normalized[-max_items:]


def build_working_memory(memory_items: list[str]) -> str:
    """将记忆条目列表构建为格式化的工作记忆字符串

    先对记忆条目进行压缩处理（去除空项并限制数量），然后将有效条目转换为
    带项目符号的列表形式字符串，每个条目以"- "开头并独占一行。

    Args:
        memory_items: 原始记忆条目字符串列表，可能包含空字符串

    Returns:
        str: 格式化后的工作记忆字符串，如果压缩后无有效条目则返回空字符串。
        例如：
        - event:query | status=success | attempt=1
        - event:search | status=failed | attempt=2
    """
    compacted = compact_short_term_memory(memory_items)
    if not compacted:
        return ""
    return "\n".join(f"- {item}" for item in compacted)
