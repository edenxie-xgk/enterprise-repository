import os
import unittest
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
}

for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)

from core.settings import settings
from src.memory.service import DisabledMemoryStore, MemoryService
from src.memory.store.base import BaseMemoryStore
from src.memory.writeback import write_long_term_memory
from src.types.memory_type import MemoryRecallQuery, MemoryRecord, MemoryWriteRequest


class FakeMemoryStore(BaseMemoryStore):
    backend = "fake"

    def __init__(self):
        self.records = {}

    def is_available(self) -> bool:
        return True

    def search(self, query: MemoryRecallQuery, query_vector: list[float]) -> list[MemoryRecord]:
        return []

    def upsert(self, record: MemoryRecord, vector: list[float]) -> str:
        self.records[record.memory_id] = record
        return record.memory_id

    def get_by_dedupe_key(self, *, user_id: str, dedupe_key: str) -> MemoryRecord | None:
        for record in self.records.values():
            if record.user_id == user_id and record.dedupe_key == dedupe_key:
                return record
        return None

    def touch(self, memory_ids: list[str], accessed_at: str) -> int:
        return 0


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

    def test_writeback_persists_explicit_memory_request(self):
        fake_store = FakeMemoryStore()
        fake_service = MemoryService(store=fake_store)

        with (
            patch.object(settings, "memory_enabled", True),
            patch.object(settings, "memory_write_enabled", True),
            patch("src.memory.writeback.memory_service", fake_service),
            patch.object(fake_service, "embed_text", return_value=[0.1, 0.2, 0.3]),
        ):
            result = write_long_term_memory(
                MemoryWriteRequest(
                    user_id="7",
                    session_id="session-1",
                    query="请记住 我在做新能源行业研究",
                    answer="好的，我会记住这个长期背景。",
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
        self.assertIn("新能源行业研究", saved_record.summary)


if __name__ == "__main__":
    unittest.main()
