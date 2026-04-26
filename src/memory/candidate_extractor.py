from __future__ import annotations

import hashlib
import re

from src.types.memory_type import MemoryWriteCandidate, MemoryWriteRequest


EXPLICIT_MEMORY_PREFIXES = (
    "\u8bb0\u4f4f",
    "\u8bf7\u8bb0\u4f4f",
    "\u5e2e\u6211\u8bb0\u4f4f",
    "\u4f60\u8bb0\u4e00\u4e0b",
    "\u8bb0\u4e00\u4e0b",
    "remember that",
    "please remember",
)

EXPLICIT_MEMORY_INFIX_MARKERS = (
    "\u8bf7\u8bb0\u4f4f",
    "\u5e2e\u6211\u8bb0\u4f4f",
    "\u8bb0\u4f4f\u8fd9\u4e2a\u4fe1\u606f",
    "\u4f60\u8bb0\u4e00\u4e0b",
    "\u8bb0\u4e00\u4e0b",
    "remember that",
    "please remember",
)

EXPLICIT_MEMORY_FILLER_PHRASES = (
    "\u8fd9\u4e2a\u4fe1\u606f",
    "\u8fd9\u6761\u4fe1\u606f",
    "\u8fd9\u4ef6\u4e8b",
    "\u8fd9\u70b9",
    "\u4e00\u4e0b",
    "this information",
    "this message",
    "this",
)


PREFERENCE_RULES = [
    (
        r"(\u4ee5\u540e|\u4eca\u540e|\u9ed8\u8ba4|by default|from now on).*(\u8be6\u7ec6|\u8be6\u7ec6\u4e00\u70b9|\u66f4\u8be6\u7ec6|detailed|more detailed)",
        "preference",
        "\u7528\u6237\u504f\u597d\u66f4\u8be6\u7ec6\u7684\u56de\u7b54",
        ["answer_style", "detailed"],
    ),
    (
        r"(\u4ee5\u540e|\u4eca\u540e|\u9ed8\u8ba4|by default|from now on).*(\u7b80\u6d01|\u7b80\u77ed|\u7cbe\u7b80|concise|brief)",
        "preference",
        "\u7528\u6237\u504f\u597d\u66f4\u7b80\u6d01\u7684\u56de\u7b54",
        ["answer_style", "concise"],
    ),
    (
        r"(\u4ee5\u540e|\u4eca\u540e|\u9ed8\u8ba4|by default|from now on).*(\u4e2d\u6587|\u4f7f\u7528\u4e2d\u6587|in chinese|speak chinese)",
        "constraint",
        "\u9ed8\u8ba4\u4f7f\u7528\u4e2d\u6587\u56de\u7b54",
        ["language", "zh-CN"],
    ),
    (
        r"(\u4ee5\u540e|\u4eca\u540e|\u9ed8\u8ba4|by default|from now on).*(\u82f1\u6587|in english|english)",
        "constraint",
        "\u9ed8\u8ba4\u4f7f\u7528\u82f1\u6587\u56de\u7b54",
        ["language", "en"],
    ),
    (
        r"(\u4e0d\u8981|\u522b|disable|do not use|don't use).*(\u8054\u7f51|web search|web\u641c\u7d22)",
        "constraint",
        "\u9ed8\u8ba4\u4e0d\u8981\u4f7f\u7528\u8054\u7f51\u641c\u7d22",
        ["web_search", "disabled"],
    ),
]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _build_dedupe_key(memory_type: str, content: str) -> str:
    normalized = _normalize_text(content).lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{memory_type}:{digest}"


def _strip_explicit_memory_fillers(text: str) -> str:
    cleaned = _normalize_text(text).strip(" :\uff1a\uff0c,\u3002\uff1f?\uff01!锛?")
    if not cleaned:
        return ""

    while cleaned:
        original = cleaned
        lowered = cleaned.lower()
        for phrase in EXPLICIT_MEMORY_FILLER_PHRASES:
            lowered_phrase = phrase.lower()
            if lowered.startswith(lowered_phrase):
                cleaned = cleaned[len(phrase) :].lstrip(" :\uff1a\uff0c,\u3002\uff1f?\uff01!锛?")
                break
            if lowered.endswith(lowered_phrase):
                cleaned = cleaned[: len(cleaned) - len(phrase)].rstrip(" :\uff1a\uff0c,\u3002\uff1f?\uff01!锛?")
                break
        if cleaned == original:
            break

    return _normalize_text(cleaned).strip(" :\uff1a\uff0c,\u3002\uff1f?\uff01!锛?")


def _extract_explicit_memory_content(query: str) -> str:
    normalized = _normalize_text(query)
    lowered = normalized.lower()
    for prefix in EXPLICIT_MEMORY_PREFIXES:
        lowered_prefix = prefix.lower()
        if lowered.startswith(lowered_prefix):
            return _strip_explicit_memory_fillers(normalized[len(prefix) :])

    for marker in EXPLICIT_MEMORY_INFIX_MARKERS:
        lowered_marker = marker.lower()
        index = lowered.find(lowered_marker)
        if index < 0:
            continue

        before = _strip_explicit_memory_fillers(normalized[:index])
        after = _strip_explicit_memory_fillers(normalized[index + len(marker) :])

        if before and index > 0:
            return before
        if after:
            return after
        if before:
            return before

    return ""


def extract_memory_write_candidates(request: MemoryWriteRequest) -> list[MemoryWriteCandidate]:
    query = _normalize_text(request.query)
    if not query:
        return []

    candidates: list[MemoryWriteCandidate] = []

    explicit_content = _extract_explicit_memory_content(query)
    if explicit_content:
        candidates.append(
            MemoryWriteCandidate(
                memory_type="task_context",
                scope="user",
                content=explicit_content,
                summary=explicit_content,
                tags=["explicit_memory"],
                importance=0.9,
                confidence=0.95,
                source="user_explicit",
                dedupe_key=_build_dedupe_key("task_context", explicit_content),
            )
        )

    for pattern, memory_type, summary, tags in PREFERENCE_RULES:
        if not re.search(pattern, query, flags=re.IGNORECASE):
            continue
        candidates.append(
            MemoryWriteCandidate(
                memory_type=memory_type,
                scope="user",
                content=summary,
                summary=summary,
                tags=tags,
                importance=0.85,
                confidence=0.9,
                source="user_explicit",
                dedupe_key=_build_dedupe_key(memory_type, summary),
            )
        )

    unique_candidates: list[MemoryWriteCandidate] = []
    seen = set()
    for candidate in candidates:
        if candidate.dedupe_key in seen:
            continue
        seen.add(candidate.dedupe_key)
        unique_candidates.append(candidate)
    return unique_candidates
