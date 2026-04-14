from __future__ import annotations

from abc import ABC, abstractmethod

from src.types.memory_type import MemoryRecord, MemoryRecallQuery


class BaseMemoryStore(ABC):
    backend: str = "disabled"

    @abstractmethod
    def is_available(self) -> bool:
        """返回底层存储是否已就绪可用。"""

    @abstractmethod
    def search(self, query: MemoryRecallQuery, query_vector: list[float]) -> list[MemoryRecord]:
        """通过向量和标量过滤器搜索记忆。"""

    @abstractmethod
    def upsert(self, record: MemoryRecord, vector: list[float]) -> str:
        """插入或更新一条记忆记录，并返回其 ID。"""

    def upsert_many(self, records: list[MemoryRecord], vectors: list[list[float]]) -> list[str]:
        if len(records) != len(vectors):
            raise ValueError("records 和 vectors 必须具有相同的长度")
        return [self.upsert(record, vector) for record, vector in zip(records, vectors)]

    @abstractmethod
    def get_by_dedupe_key(self, *, user_id: str, dedupe_key: str) -> MemoryRecord | None:
        """通过用户范围内的去重键获取已存在的记忆。"""

    @abstractmethod
    def touch(self, memory_ids: list[str], accessed_at: str) -> int:
        """更新被召回记忆的访问时间。"""
