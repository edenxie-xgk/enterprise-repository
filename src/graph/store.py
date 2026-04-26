from __future__ import annotations

from typing import Iterable

from pymongo import ASCENDING, DESCENDING

from core.settings import settings
from src.database.mongodb import mongodb_client
from src.graph.ranking import score_graph_fact, select_diverse_facts
from src.types.graph_type import FinancialFact, GraphEntity, GraphQueryContext
from utils.utils import get_current_time


class FinancialGraphStore:
    def __init__(self) -> None:
        self.entities = mongodb_client.get_collection(settings.graph_entity_collection_name)
        self.facts = mongodb_client.get_collection(settings.graph_fact_collection_name)
        self._indexes_ready = False

    def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        self.entities.create_index([("entity_id", ASCENDING)], unique=True)
        self.entities.create_index([("normalized_name", ASCENDING)])
        self.entities.create_index([("entity_type", ASCENDING), ("normalized_name", ASCENDING)])
        self.entities.create_index([("department_id", ASCENDING), ("entity_type", ASCENDING)])

        self.facts.create_index([("fact_key", ASCENDING)], unique=True)
        self.facts.create_index([("department_id", ASCENDING), ("fact_kind", ASCENDING)])
        self.facts.create_index([("normalized_metric_name", ASCENDING), ("period_year", DESCENDING)])
        self.facts.create_index([("topic", ASCENDING), ("period_year", DESCENDING)])
        self.facts.create_index([("normalized_company_name", ASCENDING), ("period_year", DESCENDING)])
        self.facts.create_index([("search_terms", ASCENDING)])
        self.facts.create_index([("period_year", DESCENDING), ("confidence", DESCENDING)])
        self._indexes_ready = True

    def upsert_entities(self, entities: Iterable[GraphEntity]) -> int:
        self.ensure_indexes()
        count = 0
        now = get_current_time()
        for entity in entities:
            payload = entity.model_dump()
            payload["updated_at"] = now
            result = self.entities.update_one(
                {"entity_id": entity.entity_id},
                {"$set": payload, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            count += int(bool(result.upserted_id or result.modified_count or result.matched_count))
        return count

    def upsert_facts(self, facts: Iterable[FinancialFact]) -> int:
        self.ensure_indexes()
        count = 0
        now = get_current_time()
        for fact in facts:
            payload = fact.model_dump()
            payload["updated_at"] = now
            result = self.facts.update_one(
                {"fact_key": fact.fact_key},
                {"$set": payload, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            count += int(bool(result.upserted_id or result.modified_count or result.matched_count))
        return count

    def search_facts(self, query_context: GraphQueryContext) -> list[FinancialFact]:
        self.ensure_indexes()
        mongo_filter = self._build_filter(query_context)
        limit = max(query_context.max_candidate_facts * 3, query_context.top_k * 4)
        rows = list(self.facts.find(mongo_filter).limit(limit))

        if not rows and query_context.filters:
            fallback_filter = self._build_department_filter(query_context)
            rows = list(self.facts.find(fallback_filter).limit(limit))

        if not rows:
            return []

        ranked = sorted(rows, key=lambda item: score_graph_fact(item, query_context), reverse=True)
        candidate_rows = ranked[: max(query_context.max_candidate_facts, query_context.top_k * 2)]
        top_rows = select_diverse_facts(candidate_rows, query_context, query_context.top_k)
        return [FinancialFact.model_validate(row) for row in top_rows]

    @staticmethod
    def _build_department_filter(query_context: GraphQueryContext) -> dict:
        filters = query_context.filters or {}
        department_id = filters.get("department_id")
        if department_id is None:
            return {}
        if isinstance(department_id, (list, tuple, set)):
            return {"department_id": {"$in": list(department_id)}}
        return {"department_id": department_id}

    def _build_filter(self, query_context: GraphQueryContext) -> dict:
        mongo_filter = self._build_department_filter(query_context)
        or_clauses: list[dict] = []

        if query_context.metric_names:
            or_clauses.append({"normalized_metric_name": {"$in": list(query_context.metric_names)}})
        if query_context.topics:
            or_clauses.append({"topic": {"$in": list(query_context.topics)}})
        if query_context.years:
            or_clauses.append({"period_year": {"$in": list(query_context.years)}})
        if query_context.company_terms:
            or_clauses.append({"normalized_company_name": {"$in": [item.lower() for item in query_context.company_terms]}})
        if query_context.search_terms:
            or_clauses.append({"search_terms": {"$in": list(query_context.search_terms[:20])}})

        if or_clauses:
            mongo_filter["$or"] = or_clauses
        return mongo_filter
