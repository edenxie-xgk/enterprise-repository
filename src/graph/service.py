from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.graph.extractor import FinancialFactExtractor
from src.graph.generator import generate_financial_graph_answer
from src.graph.planner import interpret_financial_graph_query
from src.models.llm import chatgpt_llm
from src.types.graph_type import FinancialFact, GraphBuildResult, GraphQueryContext, GraphQueryInterpretation
from src.types.rag_type import DocumentInfo, RAGResult

if TYPE_CHECKING:
    from src.graph.store import FinancialGraphStore


def _dedupe_queries(queries: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for query in queries:
        normalized = " ".join((query or "").strip().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_citations(raw_citations, allowed_citations: list[str]) -> list[str]:
    allowed_set = {item for item in allowed_citations if item}
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_citations or []:
        citation = str(item).strip()
        if not citation or citation not in allowed_set or citation in seen:
            continue
        seen.add(citation)
        normalized.append(citation)
    return normalized


class FinancialGraphService:
    def __init__(
        self,
        *,
        store: "FinancialGraphStore" | None = None,
        extractor: FinancialFactExtractor | None = None,
    ) -> None:
        self._store = store
        self.extractor = extractor or FinancialFactExtractor()

    @property
    def store(self) -> "FinancialGraphStore":
        if self._store is None:
            from src.graph.store import FinancialGraphStore

            self._store = FinancialGraphStore()
        return self._store

    @staticmethod
    def is_enabled() -> bool:
        return bool(settings.graph_enabled)

    def ingest_chunk(self, *, node_id: str, text: str, metadata: DocumentMetadata) -> GraphBuildResult:
        if not self.is_enabled():
            return GraphBuildResult(success=True, message="graph ingest skipped", diagnostics=["graph_disabled"])

        bundle = self.extractor.extract_chunk(node_id=node_id, text=text, metadata=metadata)
        entity_count = self.store.upsert_entities(bundle.entities)
        fact_count = self.store.upsert_facts(bundle.facts[: settings.graph_max_facts_per_chunk])
        return GraphBuildResult(
            success=True,
            message="graph ingest completed",
            entity_count=entity_count,
            fact_count=fact_count,
            diagnostics=bundle.diagnostics,
        )

    def ingest_nodes(self, nodes: Iterable) -> GraphBuildResult:
        if not self.is_enabled():
            return GraphBuildResult(success=True, message="graph ingest skipped", diagnostics=["graph_disabled"])

        total_entities = 0
        total_facts = 0
        diagnostics: list[str] = []
        for node in nodes or []:
            node_id = getattr(node, "id_", None) or getattr(node, "node_id", None)
            text = getattr(node, "text", "")
            raw_metadata = getattr(node, "metadata", {}) or {}
            metadata = raw_metadata if isinstance(raw_metadata, DocumentMetadata) else DocumentMetadata(**raw_metadata)
            try:
                result = self.ingest_chunk(node_id=node_id, text=text, metadata=metadata)
                total_entities += result.entity_count
                total_facts += result.fact_count
                diagnostics.extend(result.diagnostics)
            except Exception as exc:
                diagnostics.append(f"graph_chunk_ingest_failed:{node_id}:{exc}")

        diagnostics.append(f"graph_total_entities={total_entities}")
        diagnostics.append(f"graph_total_facts={total_facts}")
        return GraphBuildResult(
            success=True,
            message="graph ingest completed",
            entity_count=total_entities,
            fact_count=total_facts,
            diagnostics=diagnostics,
        )

    def build_query_context(
        self,
        *,
        query: str,
        rewritten_query: str = "",
        expand_query=None,
        decompose_query=None,
        filters=None,
    ) -> GraphQueryContext:
        search_queries = _dedupe_queries(
            [
                query,
                rewritten_query,
                *(expand_query or []),
                *(decompose_query or []),
            ]
        )
        combined_text = " ".join(search_queries)
        interpretation = self.interpret_query(combined_text)
        return GraphQueryContext(
            query=query,
            rewritten_query=rewritten_query,
            expand_query=list(expand_query or []),
            decompose_query=list(decompose_query or []),
            filters=dict(filters or {}),
            top_k=settings.graph_query_top_k,
            max_candidate_facts=settings.graph_query_max_candidates,
            query_kind=interpretation.query_kind,
            metric_names=interpretation.metric_names,
            topics=interpretation.topics,
            years=interpretation.years,
            company_terms=interpretation.company_terms,
            search_terms=interpretation.search_terms,
            comparison_mode=interpretation.comparison_mode,
        )

    def query(self, context: GraphQueryContext, user_context: dict | None = None) -> RAGResult:
        del user_context
        if not self.is_enabled():
            return self._build_result(
                answer="Financial fact graph is disabled.",
                is_sufficient=False,
                fail_reason="no_data",
                retrieval_queries=[],
                diagnostics=["graph_disabled"],
                success=False,
            )

        retrieval_queries = _dedupe_queries(
            [
                context.query or "",
                context.rewritten_query or "",
                *(context.expand_query or []),
                *(context.decompose_query or []),
            ]
        )
        facts = self.store.search_facts(context)
        if not facts:
            return self._build_result(
                answer="No matching financial facts were found for the current query.",
                is_sufficient=False,
                fail_reason="no_data",
                retrieval_queries=retrieval_queries,
                diagnostics=["graph_no_matching_facts", f"graph_query_kind={context.query_kind}"],
            )

        evidence_docs = self._collect_evidence_docs(facts)
        graph_context = self._build_graph_context(facts, evidence_docs)
        allowed_citations = [doc.node_id for doc in evidence_docs if doc.node_id]
        response = generate_financial_graph_answer(
            chatgpt_llm,
            query=context.rewritten_query or context.query or "",
            context=graph_context,
            query_kind=context.query_kind,
            comparison_mode=context.comparison_mode,
            metric_names=context.metric_names,
            topics=context.topics,
            years=context.years,
            company_terms=context.company_terms,
            allowed_citations=allowed_citations,
        )

        citations = _normalize_citations(response.citations, allowed_citations)
        if not citations and len(allowed_citations) == 1:
            citations = allowed_citations[:1]

        is_sufficient = bool(response.is_sufficient)
        fail_reason = response.fail_reason
        diagnostics = [
            f"graph_query_kind={context.query_kind}",
            f"graph_fact_count={len(facts)}",
            f"graph_evidence_doc_count={len(evidence_docs)}",
        ]
        if context.metric_names:
            diagnostics.append(f"graph_metrics={','.join(context.metric_names)}")
        if context.topics:
            diagnostics.append(f"graph_topics={','.join(context.topics)}")
        if response.reason:
            diagnostics.append(f"graph_answer_reason={response.reason}")

        if not evidence_docs:
            is_sufficient = False
            fail_reason = fail_reason or "no_data"
            diagnostics.append("graph_missing_evidence_docs")
        if not (response.evidence_summary or "").strip():
            is_sufficient = False
            fail_reason = fail_reason or "insufficient_context"
            diagnostics.append("graph_empty_summary")
        if not citations:
            is_sufficient = False
            fail_reason = fail_reason or "insufficient_context"
            diagnostics.append("graph_missing_citations")

        return self._build_result(
            answer=response.answer or response.evidence_summary or "",
            documents=evidence_docs,
            citations=citations,
            evidence_summary=response.evidence_summary or "",
            is_sufficient=is_sufficient,
            fail_reason=fail_reason,
            retrieval_queries=retrieval_queries,
            retrieval_candidate_node_ids=[fact.fact_id for fact in facts],
            rerank_node_ids=[fact.fact_id for fact in facts],
            diagnostics=diagnostics + ["graph_query_completed"],
            metadata={
                "facts": [fact.model_dump() for fact in facts],
                "query_kind": context.query_kind,
            },
        )

    @staticmethod
    def interpret_query(text: str) -> GraphQueryInterpretation:
        return interpret_financial_graph_query(text)

    @staticmethod
    def _collect_evidence_docs(facts: list[FinancialFact]) -> list[DocumentInfo]:
        docs: list[DocumentInfo] = []
        seen: set[str] = set()
        for fact in facts:
            for evidence in fact.evidence_docs:
                if not evidence.node_id or evidence.node_id in seen:
                    continue
                seen.add(evidence.node_id)
                docs.append(
                    DocumentInfo(
                        node_id=evidence.node_id,
                        content=evidence.content,
                        metadata=evidence.metadata,
                    )
                )
        return docs

    @staticmethod
    def _build_graph_context(facts: list[FinancialFact], evidence_docs: list[DocumentInfo]) -> str:
        fact_blocks = []
        for fact in facts:
            fact_blocks.append(
                "\n".join(
                    [
                        f"[fact_id:{fact.fact_id}]",
                        f"kind: {fact.fact_kind}",
                        f"company: {fact.company_name}",
                        f"period_end: {fact.period_end or ''}",
                        f"period_type: {fact.period_type or ''}",
                        f"metric: {fact.metric_name or ''}",
                        f"topic: {fact.topic or ''}",
                        f"value: {fact.raw_value or ''}",
                        f"summary: {fact.summary}",
                        f"evidence_node_ids: {', '.join(fact.evidence_node_ids)}",
                    ]
                )
            )

        evidence_blocks = []
        for doc in evidence_docs:
            evidence_blocks.append(f"[node_id:{doc.node_id}]\n{doc.content}")

        return "\n\n".join(
            [
                "[Financial Fact Graph]",
                "\n\n".join(fact_blocks),
                "[Evidence Chunks]",
                "\n\n".join(evidence_blocks),
            ]
        )

    @staticmethod
    def _build_result(
        *,
        answer: str,
        documents: list[DocumentInfo] | None = None,
        citations: list[str] | None = None,
        evidence_summary: str | None = None,
        is_sufficient: bool,
        fail_reason=None,
        retrieval_queries: list[str] | None = None,
        retrieval_candidate_node_ids: list[str] | None = None,
        rerank_node_ids: list[str] | None = None,
        diagnostics: list[str] | None = None,
        metadata: dict | None = None,
        success: bool = True,
    ) -> RAGResult:
        return RAGResult(
            success=success,
            name="graph_rag",
            answer=answer,
            evidence_summary=evidence_summary if evidence_summary is not None else answer,
            documents=documents or [],
            citations=citations or [],
            is_sufficient=is_sufficient,
            fail_reason=fail_reason,
            retrieval_queries=retrieval_queries or [],
            retrieval_candidate_node_ids=retrieval_candidate_node_ids or [],
            rerank_node_ids=rerank_node_ids or [],
            diagnostics=diagnostics or [],
            metadata=metadata or {},
        )


graph_service = FinancialGraphService()
