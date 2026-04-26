from __future__ import annotations

from core.settings import settings
from src.memory import memory_service
from src.memory.candidate_extractor import extract_memory_write_candidates
from src.types.memory_type import MemoryWriteRequest, MemoryWriteResult


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

    candidates = extract_memory_write_candidates(request)
    if not candidates:
        return MemoryWriteResult(
            success=True,
            message="memory write skipped",
            diagnostics=["memory_write_no_candidates"],
        )

    skipped_count = 0
    records_to_write = []

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
        records_to_write.append(record)

    memory_ids = memory_service.save_records(records_to_write)
    written_count = len(memory_ids)

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
