import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_TEST_ENV = {
    "DELETE_FILE": "false",
    "DATABASE_NAME": "test_db",
    "DATABASE_STRING": "postgresql://user:pass@localhost:5432/test_db",
    "DATABASE_ASYNC_STRING": "postgresql+asyncpg://user:pass@localhost:5432/test_db",
    "VECTOR_TABLE_NAME": "vectors",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "EMBEDDING_DIM": "1536",
    "MONGODB_URL": "mongodb://localhost:27017",
    "MONGODB_DB_NAME": "test_db",
    "DOC_COLLECTION_NAME": "docs",
    "QA_COLLECTION_NAME": "qa",
    "ELASTICSEARCH_URL": "http://localhost:9200",
    "METADATA_VERSION": "1",
    "TXT_CHUNK_SIZE": "500",
    "TXT_CHUNK_OVERLAP": "50",
    "TXT_MIN_CHUNK_SIZE": "100",
    "DOCX_CHUNK_SIZE": "500",
    "DOCX_CHUNK_OVERLAP": "50",
    "DOCX_MIN_CHUNK_SIZE": "100",
    "MD_CHUNK_SIZE": "500",
    "MD_CHUNK_OVERLAP": "50",
    "MD_MIN_CHUNK_SIZE": "100",
    "PDF_CHUNK_SIZE": "500",
    "PDF_CHUNK_OVERLAP": "50",
    "OCR_LANG": "ch",
    "OCR_MIN_SCORE": "0.5",
    "EXCEL_CHUNK_SIZE": "500",
    "EXCEL_MIN_CHUNK_SIZE": "100",
    "EXCEL_CHUNK_OVERLAP": "50",
    "EXCEL_HEADER_MODE": "multi",
    "PPTX_CHUNK_SIZE": "500",
    "PPTX_CHUNK_OVERLAP": "50",
    "JSON_CHUNK_SIZE": "500",
    "JSON_CHUNK_OVERLAP": "50",
    "JSON_MIN_CHUNK_SIZE": "100",
    "IMAGE_CHUNK_SIZE": "500",
    "IMAGE_CHUNK_OVERLAP": "50",
    "RETRIEVER_TOP_K": "5",
    "RERANKER_TOP_K": "5",
    "RERANKER_TYPE": "llm",
    "BM25_RETRIEVAL_MODE": "lite",
    "RERANKER_MAX_LEN": "512",
    "RETRIEVAL_MIN_SCORE": "0.1",
    "RERANKER_MIN_SCORE": "0.1",
    "CONTEXT_MAX_LEN": "4000",
    "MAX_EXPAND": "3",
    "UPDATE_DOC_TIME": "60",
    "MAX_RETRIES": "1",
    "MAX_TIMEOUT": "30",
    "HF_TOKEN": "test-token",
    "RERANKER_MODEL": "test-reranker",
    "OPENAI_API_KEY": "test-openai-key",
    "OPENAI_MODEL": "gpt-4o-mini",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "DEEPSEEK_URL": "https://api.deepseek.com",
    "DEEPSEEK_MODEL": "deepseek-chat",
    "DEEPSEEK_API_KEY": "test-deepseek-key",
    "ZHIPUAI_API_KEY": "test-zhipu-key",
    "MEMORY_BACKEND": "disabled",
    "MEMORY_ENABLED": "false",
    "MEMORY_WRITE_ENABLED": "false",
    "MILVUS_VECTOR_DIM": "1536",
}

_FORCED_ENV_KEYS = {
    "EMBEDDING_DIM",
    "MEMORY_BACKEND",
    "MEMORY_ENABLED",
    "MEMORY_WRITE_ENABLED",
    "MILVUS_VECTOR_DIM",
}

for _key, _value in _TEST_ENV.items():
    if _key in _FORCED_ENV_KEYS:
        os.environ[_key] = _value
    else:
        os.environ.setdefault(_key, _value)

from core.settings import settings
from src.memory.service import DisabledMemoryStore, MemoryService
from src.memory.store.base import BaseMemoryStore
from src.memory.store.milvus_store import MilvusMemoryStore
from src.memory.writeback import write_long_term_memory
from src.types.memory_type import MemoryRecallQuery, MemoryRecord, MemoryWriteRequest


class FakeMemoryStore(BaseMemoryStore):
    backend = "fake"

    def __init__(self):
        self.records = {}
        self.upsert_calls = 0
        self.upsert_many_calls = 0

    def is_available(self) -> bool:
        return True

    def search(self, query: MemoryRecallQuery, query_vector: list[float]) -> list[MemoryRecord]:
        return []

    def upsert(self, record: MemoryRecord, vector: list[float]) -> str:
        self.upsert_calls += 1
        self.records[record.memory_id] = record
        return record.memory_id

    def upsert_many(self, records: list[MemoryRecord], vectors: list[list[float]]) -> list[str]:
        self.upsert_many_calls += 1
        for record in records:
            self.records[record.memory_id] = record
        return [record.memory_id for record in records]

    def get_by_dedupe_key(self, *, user_id: str, dedupe_key: str) -> MemoryRecord | None:
        for record in self.records.values():
            if record.user_id == user_id and record.dedupe_key == dedupe_key:
                return record
        return None

    def touch(self, memory_ids: list[str], accessed_at: str) -> int:
        return 0


class TouchFailStore(BaseMemoryStore):
    backend = "fake"

    def is_available(self) -> bool:
        return True

    def search(self, query: MemoryRecallQuery, query_vector: list[float]) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                memory_id="m1",
                user_id=query.user_id,
                session_id=query.session_id,
                scope="user",
                memory_type="preference",
                content="prefer detailed answers",
                summary="prefer detailed answers",
                tags=["answer_style", "detailed"],
                importance=0.9,
                confidence=0.95,
                source="user_explicit",
                dedupe_key="preference:test",
                created_at="2026-04-15T10:00:00",
                updated_at="2026-04-15T10:00:00",
            )
        ]

    def upsert(self, record: MemoryRecord, vector: list[float]) -> str:
        return record.memory_id

    def get_by_dedupe_key(self, *, user_id: str, dedupe_key: str) -> MemoryRecord | None:
        return None

    def touch(self, memory_ids: list[str], accessed_at: str) -> int:
        raise RuntimeError("touch failed")


class FakeMilvusDataType:
    VARCHAR = "VARCHAR"
    BOOL = "BOOL"
    FLOAT = "FLOAT"
    FLOAT_VECTOR = "FLOAT_VECTOR"


class FakeMilvusSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class FakeMilvusIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeMilvusClient:
    databases = {"default"}
    collection_exists = False
    init_calls = []
    create_database_calls = []
    create_collection_calls = []
    close_calls = 0

    @classmethod
    def reset(cls, *, databases=None, collection_exists=False):
        cls.databases = set(databases or {"default"})
        cls.collection_exists = collection_exists
        cls.init_calls = []
        cls.create_database_calls = []
        cls.create_collection_calls = []
        cls.close_calls = 0

    def __init__(self, *, uri, token=None, db_name=None, **kwargs):
        self.uri = uri
        self.token = token
        self.db_name = db_name or ""
        type(self).init_calls.append({"uri": uri, "token": token, "db_name": self.db_name})
        if self.db_name and self.db_name not in type(self).databases:
            raise RuntimeError(f"database not found[database={self.db_name}]")

    def list_databases(self, *args, **kwargs):
        return sorted(type(self).databases)

    def create_database(self, db_name, *args, **kwargs):
        type(self).create_database_calls.append(db_name)
        type(self).databases.add(db_name)

    def close(self):
        type(self).close_calls += 1

    def has_collection(self, collection_name):
        return type(self).collection_exists

    def create_schema(self, **kwargs):
        return FakeMilvusSchema()

    def prepare_index_params(self):
        return FakeMilvusIndexParams()

    def create_collection(self, **kwargs):
        type(self).create_collection_calls.append(kwargs["collection_name"])


class LongTermMemoryTests(unittest.TestCase):
    def test_memory_service_recall_skips_when_store_is_unavailable(self):
        service = MemoryService(store=DisabledMemoryStore())

        result = service.recall(
            MemoryRecallQuery(
                user_id="42",
                query="please remember my preference",
            )
        )

        self.assertTrue(result.success)
        self.assertFalse(result.used)
        self.assertIn("memory_store_unavailable", result.diagnostics)

    def test_memory_service_recall_skips_when_milvus_connection_fails(self):
        class FailingMilvusClient:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("milvus unavailable")

        service = MemoryService(store=MilvusMemoryStore())

        with (
            patch.dict("sys.modules", {"pymilvus": SimpleNamespace(MilvusClient=FailingMilvusClient)}),
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_backend", "milvus"),
            patch.object(settings, "milvus_uri", "http://127.0.0.1:19530"),
        ):
            result = service.recall(
                MemoryRecallQuery(
                    user_id="42",
                    query="please remember my preference",
                )
            )

        self.assertTrue(result.success)
        self.assertFalse(result.used)
        self.assertIn("memory_store_unavailable", result.diagnostics)
        self.assertIn("memory_store_import_error=milvus unavailable", result.diagnostics)

    def test_milvus_store_auto_creates_missing_database(self):
        FakeMilvusClient.reset(databases={"default"}, collection_exists=False)
        store = MilvusMemoryStore()

        with (
            patch.dict(
                "sys.modules",
                {"pymilvus": SimpleNamespace(MilvusClient=FakeMilvusClient, DataType=FakeMilvusDataType)},
            ),
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_backend", "milvus"),
            patch.object(settings, "milvus_uri", "127.0.0.1:19530"),
            patch.object(settings, "milvus_token", "test-token"),
            patch.object(settings, "milvus_db_name", "rag_agent"),
            patch.object(settings, "milvus_collection_name", "long_term_memory"),
            patch.object(settings, "milvus_consistency_level", "Strong"),
            patch.object(settings, "milvus_search_metric", "COSINE"),
            patch.object(settings, "milvus_index_type", "AUTOINDEX"),
            patch.object(settings, "milvus_vector_dim", 1536),
        ):
            self.assertTrue(store.is_available())

        self.assertEqual(FakeMilvusClient.create_database_calls, ["rag_agent"])
        self.assertEqual(FakeMilvusClient.create_collection_calls, ["long_term_memory"])
        self.assertEqual(len(FakeMilvusClient.init_calls), 2)
        self.assertEqual(FakeMilvusClient.init_calls[0]["uri"], "http://127.0.0.1:19530")
        self.assertEqual(FakeMilvusClient.init_calls[0]["db_name"], "")
        self.assertEqual(FakeMilvusClient.init_calls[1]["db_name"], "rag_agent")
        self.assertEqual(FakeMilvusClient.close_calls, 1)

    def test_milvus_store_skips_database_creation_when_database_exists(self):
        FakeMilvusClient.reset(databases={"default", "rag_agent"}, collection_exists=True)
        store = MilvusMemoryStore()

        with (
            patch.dict(
                "sys.modules",
                {"pymilvus": SimpleNamespace(MilvusClient=FakeMilvusClient, DataType=FakeMilvusDataType)},
            ),
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_backend", "milvus"),
            patch.object(settings, "milvus_uri", "http://127.0.0.1:19530"),
            patch.object(settings, "milvus_token", ""),
            patch.object(settings, "milvus_db_name", "rag_agent"),
            patch.object(settings, "milvus_collection_name", "long_term_memory"),
            patch.object(settings, "milvus_consistency_level", "Strong"),
            patch.object(settings, "milvus_search_metric", "COSINE"),
            patch.object(settings, "milvus_index_type", "AUTOINDEX"),
            patch.object(settings, "milvus_vector_dim", 1536),
        ):
            self.assertTrue(store.is_available())

        self.assertEqual(FakeMilvusClient.create_database_calls, [])
        self.assertEqual(FakeMilvusClient.create_collection_calls, [])
        self.assertEqual(len(FakeMilvusClient.init_calls), 2)
        self.assertEqual(FakeMilvusClient.init_calls[1]["db_name"], "rag_agent")
        self.assertEqual(FakeMilvusClient.close_calls, 1)

    def test_writeback_persists_explicit_memory_request(self):
        fake_store = FakeMemoryStore()
        fake_service = MemoryService(store=fake_store)

        with (
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_write_enabled", True),
            patch.object(settings, "memory_backend", "disabled"),
            patch("src.memory.writeback.memory_service", fake_service),
            patch.object(fake_service, "embed_text", return_value=[0.1, 0.2, 0.3]),
        ):
            result = write_long_term_memory(
                MemoryWriteRequest(
                    user_id="7",
                    session_id="session-1",
                    query="\u8bf7\u8bb0\u4f4f\uff1a\u6211\u53eb\u5c0f\u738b",
                    answer="\u597d\u7684\uff0c\u6211\u4f1a\u8bb0\u4f4f\u3002",
                    chat_history=[],
                    user_profile={},
                    existing_memories=[],
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.written_count, 1)
        self.assertEqual(len(fake_store.records), 1)
        saved_record = next(iter(fake_store.records.values()))
        self.assertEqual(saved_record.user_id, "7")
        self.assertEqual(saved_record.memory_type, "task_context")
        self.assertEqual(saved_record.summary, "\u6211\u53eb\u5c0f\u738b")
        self.assertEqual(fake_store.upsert_many_calls, 1)
        self.assertEqual(fake_store.upsert_calls, 0)

    def test_writeback_persists_explicit_memory_when_marker_appears_after_content(self):
        fake_store = FakeMemoryStore()
        fake_service = MemoryService(store=fake_store)

        with (
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_write_enabled", True),
            patch.object(settings, "memory_backend", "disabled"),
            patch("src.memory.writeback.memory_service", fake_service),
            patch.object(fake_service, "embed_text", return_value=[0.1, 0.2, 0.3]),
        ):
            result = write_long_term_memory(
                MemoryWriteRequest(
                    user_id="10",
                    session_id="session-4",
                    query="\u6211\u662f\u4e00\u540d\u7a0b\u5e8f\u5458\uff0c\u8bf7\u8bb0\u4f4f\u8fd9\u4e2a\u4fe1\u606f",
                    answer="\u597d\u7684\uff0c\u6211\u4f1a\u8bb0\u4f4f\u3002",
                    chat_history=[],
                    user_profile={},
                    existing_memories=[],
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.written_count, 1)
        saved_record = next(iter(fake_store.records.values()))
        self.assertEqual(saved_record.summary, "\u6211\u662f\u4e00\u540d\u7a0b\u5e8f\u5458")

    def test_writeback_persists_explicit_memory_when_marker_appears_before_content(self):
        fake_store = FakeMemoryStore()
        fake_service = MemoryService(store=fake_store)

        with (
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_write_enabled", True),
            patch.object(settings, "memory_backend", "disabled"),
            patch("src.memory.writeback.memory_service", fake_service),
            patch.object(fake_service, "embed_text", return_value=[0.1, 0.2, 0.3]),
        ):
            result = write_long_term_memory(
                MemoryWriteRequest(
                    user_id="11",
                    session_id="session-5",
                    query="\u8bf7\u8bb0\u4f4f\u8fd9\u4e2a\u4fe1\u606f\uff1a\u6211\u662f\u4e00\u540d\u7a0b\u5e8f\u5458",
                    answer="\u597d\u7684\uff0c\u6211\u4f1a\u8bb0\u4f4f\u3002",
                    chat_history=[],
                    user_profile={},
                    existing_memories=[],
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.written_count, 1)
        saved_record = next(iter(fake_store.records.values()))
        self.assertEqual(saved_record.summary, "\u6211\u662f\u4e00\u540d\u7a0b\u5e8f\u5458")

    def test_writeback_upserts_preference_rules_without_duplicates(self):
        fake_store = FakeMemoryStore()
        fake_service = MemoryService(store=fake_store)

        with (
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_write_enabled", True),
            patch.object(settings, "memory_backend", "disabled"),
            patch("src.memory.writeback.memory_service", fake_service),
            patch.object(fake_service, "embed_text", return_value=[0.1, 0.2, 0.3]),
        ):
            request = MemoryWriteRequest(
                user_id="8",
                session_id="session-2",
                query="\u4ee5\u540e\u9ed8\u8ba4\u7528\u4e2d\u6587\uff0c\u5e76\u4e14\u56de\u7b54\u8be6\u7ec6\u4e00\u70b9",
                answer="\u597d\u7684\uff0c\u6211\u4f1a\u6309\u8fd9\u4e2a\u504f\u597d\u56de\u7b54\u3002",
                chat_history=[],
                user_profile={},
                existing_memories=[],
            )
            first = write_long_term_memory(request)
            second = write_long_term_memory(request)

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertEqual(first.written_count, 2)
        self.assertEqual(second.written_count, 2)
        self.assertEqual(len(fake_store.records), 2)
        summaries = {record.summary for record in fake_store.records.values()}
        self.assertEqual(
            summaries,
            {
                "\u7528\u6237\u504f\u597d\u66f4\u8be6\u7ec6\u7684\u56de\u7b54",
                "\u9ed8\u8ba4\u4f7f\u7528\u4e2d\u6587\u56de\u7b54",
            },
        )
        self.assertEqual(fake_store.upsert_many_calls, 2)
        self.assertEqual(fake_store.upsert_calls, 0)

    def test_recall_still_succeeds_when_touch_fails(self):
        service = MemoryService(store=TouchFailStore())

        with (
            patch.object(settings, "memory_backend", "disabled"),
            patch.object(service, "embed_text", return_value=[0.1, 0.2, 0.3]),
        ):
            result = service.recall(
                MemoryRecallQuery(
                    user_id="9",
                    session_id="session-3",
                    query="please explain in detail",
                    top_k=3,
                    min_score=0.0,
                    scopes=["user", "session"],
                )
            )

        self.assertTrue(result.success)
        self.assertTrue(result.used)
        self.assertEqual(len(result.memories), 1)
        self.assertIn("memory_touch_failed=touch failed", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
