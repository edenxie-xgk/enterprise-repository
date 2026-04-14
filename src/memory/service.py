from __future__ import annotations

from uuid import uuid4

from core.settings import settings
from src.memory.store.base import BaseMemoryStore
from src.memory.store.milvus_store import MilvusMemoryStore
from src.types.memory_type import MemoryRecord, MemoryRecallQuery, MemoryRecallResult
from utils.utils import get_current_time


class DisabledMemoryStore(BaseMemoryStore):
    backend = "disabled"

    def is_available(self) -> bool:
        return False

    def search(self, query: MemoryRecallQuery, query_vector: list[float]) -> list[MemoryRecord]:
        return []

    def upsert(self, record: MemoryRecord, vector: list[float]) -> str:
        return record.memory_id

    def get_by_dedupe_key(self, *, user_id: str, dedupe_key: str) -> MemoryRecord | None:
        return None

    def touch(self, memory_ids: list[str], accessed_at: str) -> int:
        return 0


def _normalize_memory_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _build_memory_context(memories: list[MemoryRecord], *, limit: int) -> str:
    if not memories:
        return ""

    rows = []
    for memory in memories[:limit]:
        label = memory.memory_type.replace("_", " ")
        summary = (memory.summary or memory.content or "").strip()
        if not summary:
            continue
        rows.append(f"- [{label}] {summary}")
    return "\n".join(rows)


class MemoryService:
    def __init__(self, store: BaseMemoryStore | None = None) -> None:
        self.store = store or self._build_store()

    @staticmethod
    def _build_store() -> BaseMemoryStore:
        if settings.memory_enabled and settings.memory_backend == "milvus":
            return MilvusMemoryStore()
        return DisabledMemoryStore()

    @staticmethod
    def embed_text(text: str) -> list[float]:
        from src.models.embedding import embed_model

        normalized = _normalize_memory_text(text)
        if not normalized:
            return []
        return list(embed_model.get_text_embedding(normalized))

    def recall(self, query: MemoryRecallQuery) -> MemoryRecallResult:
        normalized_query = _normalize_memory_text(query.query)
        if not query.user_id or not normalized_query:
            return MemoryRecallResult(
                success=True,
                message="memory recall skipped",
                diagnostics=["memory_recall_skipped_empty_input"],
            )

        if not self.store.is_available():
            diagnostics = ["memory_store_unavailable"]
            import_error = getattr(self.store, "import_error", None)
            if import_error:
                diagnostics.append(f"memory_store_import_error={import_error}")
            return MemoryRecallResult(
                success=True,
                message="memory recall skipped",
                diagnostics=diagnostics,
            )

        vector = self.embed_text(normalized_query)
        if not vector:
            return MemoryRecallResult(
                success=True,
                message="memory recall skipped",
                diagnostics=["memory_recall_skipped_empty_vector"],
            )
        if settings.memory_backend == "milvus" and len(vector) != settings.milvus_vector_dim:
            return MemoryRecallResult(
                success=False,
                message="memory recall failed",
                diagnostics=[
                    f"memory_vector_dim_mismatch expected={settings.milvus_vector_dim} actual={len(vector)}",
                ],
            )

        memories = self.store.search(query, vector)
        context = _build_memory_context(memories, limit=settings.memory_context_limit)
        used = bool(context)

        if memories:
            self.store.touch(
                [memory.memory_id for memory in memories if memory.memory_id],
                accessed_at=get_current_time(),
            )

        diagnostics = [f"memory_recall_count={len(memories)}"]
        if used:
            diagnostics.append("memory_recall_context_built")

        return MemoryRecallResult(
            success=True,
            message="memory recall completed",
            memories=memories,
            memory_context=context,
            used=used,
            diagnostics=diagnostics,
        )

    def build_record(
        self,
        *,
        memory_id: str | None = None,
        user_id: str,
        session_id: str | None,
        memory_type: str,
        scope: str = "user",
        content: str,
        summary: str,
        tags: list[str],
        importance: float,
        confidence: float,
        source: str,
        dedupe_key: str,
        created_at: str | None = None,
        expires_at: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        now = get_current_time()
        return MemoryRecord(
            memory_id=memory_id or str(uuid4()),
            user_id=user_id,
            session_id=session_id,
            scope=scope,
            memory_type=memory_type,
            content=_normalize_memory_text(content),
            summary=_normalize_memory_text(summary),
            tags=list(tags or []),
            importance=importance,
            confidence=confidence,
            source=source,
            dedupe_key=_normalize_memory_text(dedupe_key),
            created_at=created_at or now,
            updated_at=now,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
        )

    def save_record(self, record: MemoryRecord) -> str:
        if not self.store.is_available():
            raise RuntimeError("Memory store is not available")
        vector = self.embed_text(record.summary or record.content)
        if not vector:
            raise ValueError("Memory record cannot be embedded")
        if settings.memory_backend == "milvus" and len(vector) != settings.milvus_vector_dim:
            raise ValueError(
                f"Memory vector dim mismatch: expected {settings.milvus_vector_dim}, got {len(vector)}"
            )
        return self.store.upsert(record, vector)


memory_service = MemoryService()
