from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from core.settings import settings
from src.memory.store.base import BaseMemoryStore
from src.types.memory_type import MemoryRecord, MemoryRecallQuery


class MilvusMemoryStore(BaseMemoryStore):
    backend = "milvus"

    def __init__(self) -> None:
        self._client = None
        self._import_error: str | None = None
        self._last_error: str | None = None

    def is_available(self) -> bool:
        if not settings.memory_enabled or settings.memory_backend != "milvus":
            return False
        if not settings.milvus_uri:
            return False
        return self._ensure_client() is not None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if self._import_error is not None:
            return None

        try:
            from pymilvus import MilvusClient
        except Exception as exc:  # pragma: no cover
            self._import_error = str(exc)
            return None

        try:
            self._ensure_database_exists(MilvusClient)
            self._client = MilvusClient(**self._client_kwargs(db_name=self._target_db_name()))
            self._ensure_collection()
            self._last_error = None
            return self._client
        except Exception as exc:
            self._client = None
            self._last_error = str(exc)
            return None

    def _client_kwargs(self, *, db_name: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"uri": self._normalized_uri(settings.milvus_uri)}
        token = (settings.milvus_token or "").strip()
        if token:
            kwargs["token"] = token
        if db_name:
            kwargs["db_name"] = db_name
        return kwargs

    @staticmethod
    def _close_client(client: Any) -> None:
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            with suppress(Exception):
                close()

    @staticmethod
    def _target_db_name() -> str | None:
        raw_name = (settings.milvus_db_name or "").strip()
        return raw_name or None

    def _ensure_database_exists(self, client_cls: Any) -> None:
        target_db_name = self._target_db_name()
        if not target_db_name:
            return

        admin_client = None
        try:
            admin_client = client_cls(**self._client_kwargs())
            existing_databases = set(admin_client.list_databases() or [])
            if target_db_name in existing_databases:
                return

            try:
                admin_client.create_database(target_db_name)
            except Exception:
                refreshed_databases = set(admin_client.list_databases() or [])
                if target_db_name not in refreshed_databases:
                    raise
        finally:
            self._close_client(admin_client)

    @staticmethod
    def _normalized_uri(uri: str) -> str:
        raw = (uri or "").strip()
        if not raw:
            return raw
        if "://" in raw:
            return raw
        return f"http://{raw}"

    def _ensure_collection(self) -> None:
        client = self._client
        if client is None:
            return
        if client.has_collection(collection_name=settings.milvus_collection_name):
            return

        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="memory_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=128, nullable=True)
        schema.add_field(field_name="scope", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="memory_type", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="tags_json", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="dedupe_key", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="updated_at", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="last_accessed_at", datatype=DataType.VARCHAR, max_length=64, nullable=True)
        schema.add_field(field_name="expires_at", datatype=DataType.VARCHAR, max_length=64, nullable=True)
        schema.add_field(field_name="is_active", datatype=DataType.BOOL)
        schema.add_field(field_name="importance", datatype=DataType.FLOAT)
        schema.add_field(field_name="confidence", datatype=DataType.FLOAT)
        schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=settings.milvus_vector_dim)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type=settings.milvus_index_type,
            metric_type=settings.milvus_search_metric,
        )

        client.create_collection(
            collection_name=settings.milvus_collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level=settings.milvus_consistency_level,
        )

    @staticmethod
    def _quote(value: str) -> str:
        escaped = (value or "").replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _build_filter(self, query: MemoryRecallQuery) -> str:
        clauses = [f"user_id == {self._quote(query.user_id)}", "is_active == true"]

        scopes = [scope for scope in query.scopes if scope]
        if scopes:
            scope_clause = " or ".join(f"scope == {self._quote(scope)}" for scope in scopes)
            clauses.append(f"({scope_clause})")

        if query.memory_types:
            type_clause = " or ".join(
                f"memory_type == {self._quote(memory_type)}" for memory_type in query.memory_types
            )
            clauses.append(f"({type_clause})")

        return " and ".join(clauses)

    @staticmethod
    def _loads_json(raw_value: Any, default):
        if raw_value in (None, ""):
            return default
        if isinstance(raw_value, (list, dict)):
            return raw_value
        try:
            return json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _output_fields(*, include_vector: bool = False) -> list[str]:
        fields = [
            "memory_id",
            "user_id",
            "session_id",
            "scope",
            "memory_type",
            "content",
            "summary",
            "tags_json",
            "source",
            "dedupe_key",
            "created_at",
            "updated_at",
            "last_accessed_at",
            "expires_at",
            "is_active",
            "importance",
            "confidence",
            "metadata_json",
        ]
        if include_vector:
            fields.append("vector")
        return fields

    def _payload_from_record(self, record: MemoryRecord, vector: list[float]) -> dict[str, Any]:
        return {
            "memory_id": record.memory_id,
            "user_id": record.user_id,
            "session_id": record.session_id,
            "scope": record.scope,
            "memory_type": record.memory_type,
            "content": record.content,
            "summary": record.summary,
            "tags_json": json.dumps(record.tags, ensure_ascii=False),
            "source": record.source,
            "dedupe_key": record.dedupe_key,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "last_accessed_at": record.last_accessed_at,
            "expires_at": record.expires_at,
            "is_active": record.is_active,
            "importance": float(record.importance),
            "confidence": float(record.confidence),
            "metadata_json": json.dumps(record.metadata, ensure_ascii=False),
            "vector": vector,
        }

    def _upsert_payloads(self, payloads: list[dict[str, Any]], *, flush: bool) -> None:
        client = self._ensure_client()
        if client is None:
            raise RuntimeError("Milvus memory store is not available")
        if not payloads:
            return

        client.upsert(collection_name=settings.milvus_collection_name, data=payloads)
        if flush:
            client.flush(collection_name=settings.milvus_collection_name)

    def _record_from_hit(self, hit: dict[str, Any]) -> MemoryRecord:
        entity = hit.get("entity") or hit
        score = hit.get("distance")
        if score is None:
            score = hit.get("score")

        return MemoryRecord(
            memory_id=str(entity.get("memory_id") or ""),
            user_id=str(entity.get("user_id") or ""),
            session_id=(entity.get("session_id") or None),
            scope=str(entity.get("scope") or "user"),
            memory_type=str(entity.get("memory_type") or "task_context"),
            content=str(entity.get("content") or ""),
            summary=str(entity.get("summary") or ""),
            tags=list(self._loads_json(entity.get("tags_json"), [])),
            importance=float(entity.get("importance") or 0.0),
            confidence=float(entity.get("confidence") or 0.0),
            source=str(entity.get("source") or "assistant_extract"),
            dedupe_key=str(entity.get("dedupe_key") or ""),
            created_at=str(entity.get("created_at") or ""),
            updated_at=str(entity.get("updated_at") or ""),
            last_accessed_at=entity.get("last_accessed_at") or None,
            expires_at=entity.get("expires_at") or None,
            is_active=bool(entity.get("is_active", True)),
            score=float(score) if score is not None else None,
            metadata=dict(self._loads_json(entity.get("metadata_json"), {})),
        )

    def search(self, query: MemoryRecallQuery, query_vector: list[float]) -> list[MemoryRecord]:
        client = self._ensure_client()
        if client is None:
            return []

        try:
            results = client.search(
                collection_name=settings.milvus_collection_name,
                data=[query_vector],
                anns_field="vector",
                filter=self._build_filter(query),
                limit=query.top_k,
                consistency_level=settings.milvus_consistency_level,
                output_fields=self._output_fields(),
            )
        except Exception as exc:
            self._client = None
            self._last_error = str(exc)
            return []

        hits = results[0] if results and isinstance(results[0], list) else results
        memories = []
        for hit in hits or []:
            record = self._record_from_hit(hit)
            if record.score is not None and record.score < query.min_score:
                continue
            memories.append(record)
        return memories

    def upsert(self, record: MemoryRecord, vector: list[float]) -> str:
        return self.upsert_many([record], [vector])[0]

    def upsert_many(self, records: list[MemoryRecord], vectors: list[list[float]]) -> list[str]:
        if len(records) != len(vectors):
            raise ValueError("records and vectors must have the same length")
        if not records:
            return []

        payloads = [self._payload_from_record(record, vector) for record, vector in zip(records, vectors)]
        self._upsert_payloads(payloads, flush=True)
        return [record.memory_id for record in records]

    def get_by_dedupe_key(self, *, user_id: str, dedupe_key: str) -> MemoryRecord | None:
        client = self._ensure_client()
        if client is None or not user_id or not dedupe_key:
            return None

        expr = f"user_id == {self._quote(user_id)} and dedupe_key == {self._quote(dedupe_key)}"
        try:
            results = client.query(
                collection_name=settings.milvus_collection_name,
                filter=expr,
                consistency_level=settings.milvus_consistency_level,
                output_fields=self._output_fields(),
            )
        except Exception as exc:
            self._client = None
            self._last_error = str(exc)
            return None
        if not results:
            return None
        return self._record_from_hit(results[0])

    def touch(self, memory_ids: list[str], accessed_at: str) -> int:
        client = self._ensure_client()
        if client is None or not memory_ids:
            return 0

        try:
            rows = client.query(
                collection_name=settings.milvus_collection_name,
                ids=memory_ids,
                consistency_level=settings.milvus_consistency_level,
                output_fields=self._output_fields(include_vector=True),
            )
        except Exception as exc:
            self._client = None
            self._last_error = str(exc)
            return 0
        if not rows:
            return 0

        records: list[MemoryRecord] = []
        vectors: list[list[float]] = []
        for row in rows:
            raw_row = dict(row)
            vector = raw_row.pop("vector", None)
            if not vector:
                continue

            record = self._record_from_hit(raw_row)
            record.last_accessed_at = accessed_at
            record.updated_at = accessed_at
            records.append(record)
            vectors.append(vector)

        if not records:
            return 0

        payloads = [self._payload_from_record(record, vector) for record, vector in zip(records, vectors)]
        self._upsert_payloads(payloads, flush=False)
        return len(records)

    @property
    def import_error(self) -> str | None:
        return self._import_error or self._last_error
