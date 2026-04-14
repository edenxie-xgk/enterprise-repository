from __future__ import annotations

import re

from core.settings import settings
from src.memory import memory_service
from src.types.memory_type import MemoryWriteCandidate, MemoryWriteRequest, MemoryWriteResult


EXPLICIT_MEMORY_PREFIXES = (
    "记住",
    "请记住",
    "帮我记住",
    "你记一下",
    "记一下",
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _build_dedupe_key(memory_type: str, content: str) -> str:
    normalized = _normalize_text(content).lower()
    return f"{memory_type}:{normalized}"


def _extract_explicit_memory_content(query: str) -> str:
    normalized = _normalize_text(query)
    for prefix in EXPLICIT_MEMORY_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip(" :：，,。")
    return ""


def _extract_candidates(request: MemoryWriteRequest) -> list[MemoryWriteCandidate]:
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

    preference_rules = [
        (r"(以后|今后|默认).*(详细|详细一点)", "preference", "用户偏好更详细的回答", ["answer_style", "detailed"]),
        (r"(以后|今后|默认).*(简洁|简短|精简)", "preference", "用户偏好更简洁的回答", ["answer_style", "concise"]),
        (r"(以后|今后|默认).*(中文回答|用中文|中文回复)", "constraint", "默认使用中文回答", ["language", "zh-CN"]),
        (r"(以后|今后|默认).*(英文回答|用英文|英文回复)", "constraint", "默认使用英文回答", ["language", "en"]),
        (r"(不要|别).*(联网|web search|web搜索)", "constraint", "默认不要使用联网搜索", ["web_search", "disabled"]),
    ]

    for pattern, memory_type, summary, tags in preference_rules:
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


def write_long_term_memory(request: MemoryWriteRequest) -> MemoryWriteResult:
    if not settings.memory_enabled or not settings.memory_write_enabled:
        return MemoryWriteResult(
            success=True,
            message="memory write skipped",
            diagnostics=["memory_write_disabled"],
        )

    if not request.user_id:
        return MemoryWriteResult(
            success=True,
            message="memory write skipped",
            diagnostics=["memory_write_skipped_missing_user_id"],
        )

    if not memory_service.store.is_available():
        diagnostics = ["memory_store_unavailable_for_write"]
        import_error = getattr(memory_service.store, "import_error", None)
        if import_error:
            diagnostics.append(f"memory_store_import_error={import_error}")
        return MemoryWriteResult(
            success=True,
            message="memory write skipped",
            diagnostics=diagnostics,
        )

    candidates = _extract_candidates(request)
    if not candidates:
        return MemoryWriteResult(
            success=True,
            message="memory write skipped",
            diagnostics=["memory_write_no_candidates"],
        )

    written_count = 0
    skipped_count = 0
    memory_ids: list[str] = []

    for candidate in candidates:
        if candidate.importance < settings.memory_write_min_importance:
            skipped_count += 1
            continue

        existing = memory_service.store.get_by_dedupe_key(
            user_id=request.user_id,
            dedupe_key=candidate.dedupe_key,
        )

        record = memory_service.build_record(
            memory_id=existing.memory_id if existing else None,
            user_id=request.user_id,
            session_id=request.session_id,
            memory_type=candidate.memory_type,
            scope=candidate.scope,
            content=candidate.content,
            summary=candidate.summary,
            tags=candidate.tags,
            importance=candidate.importance,
            confidence=candidate.confidence,
            source=candidate.source,
            dedupe_key=candidate.dedupe_key,
            created_at=existing.created_at if existing else None,
            expires_at=candidate.expires_at,
            metadata=candidate.metadata,
        )
        memory_id = memory_service.save_record(record)
        memory_ids.append(memory_id)
        written_count += 1

    diagnostics = [f"memory_write_candidates={len(candidates)}", f"memory_write_written={written_count}"]
    if skipped_count:
        diagnostics.append(f"memory_write_skipped={skipped_count}")

    return MemoryWriteResult(
        success=True,
        message="memory write completed",
        written_count=written_count,
        skipped_count=skipped_count,
        memory_ids=memory_ids,
        candidates=candidates,
        diagnostics=diagnostics,
    )
