from typing import Iterable, Literal, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.congfig.llm_config import LLMService
from src.models.llm import chatgpt_llm
from src.prompts.agent.sub_query_aggregate import SUB_QUERY_AGGREGATE_PROMPT
from src.rag.rag_service import rag_service
from src.types.rag_type import RAGResult, RagContext, SubQueryResult


DENSE_SCORE_THRESHOLD = 0.3
RERANK_SCORE_THRESHOLD = 0.5
CONFIDENCE_WEIGHT = {"dense": 0.4, "bm25": 0.3, "rerank": 0.3}
MAX_SUB_QUERIES = 3


class AggregateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_summary: Optional[str] = Field(default="", description="Aggregated evidence summary")
    is_sufficient: bool = Field(default=False, description="Whether aggregated evidence is sufficient")
    reason: Optional[str] = Field(default=None, description="Aggregation reason")
    fail_reason: Optional[Literal["insufficient_context", "ambiguous_query", "no_data"]] = Field(
        default=None,
        description="Aggregation failure reason",
    )


def compute_confidence(rag_result: RAGResult) -> float:
    if not rag_result.documents:
        return 0.0

    scores = []
    for doc in rag_result.documents:
        dense = doc.dense_score or 0.0
        bm25 = doc.bm25_score or 0.0
        rerank = doc.rerank_score or 0.0
        score = (
            dense * CONFIDENCE_WEIGHT["dense"]
            + bm25 * CONFIDENCE_WEIGHT["bm25"]
            + rerank * CONFIDENCE_WEIGHT["rerank"]
        )
        scores.append(score)

    top_scores = sorted(scores, reverse=True)[:3]
    return sum(top_scores) / len(top_scores)


def has_any_score_below(documents: Iterable, field: str, threshold: float) -> bool:
    for doc in documents:
        value = getattr(doc, field, None)
        if value is not None and value < threshold:
            return True
    return False


def decide_fail_reason(rag_result: RAGResult):
    if rag_result.is_sufficient:
        return None
    if not rag_result.documents:
        return "no_data"
    if all((doc.dense_score or 0.0) < DENSE_SCORE_THRESHOLD for doc in rag_result.documents):
        return "low_recall"
    if has_any_score_below(rag_result.documents, "rerank_score", RERANK_SCORE_THRESHOLD):
        return "bad_ranking"
    return "ambiguous_query"


def should_run_multi_pass(query: RagContext) -> bool:
    sub_queries = [item.strip() for item in query.decompose_query if item and item.strip()]
    return 2 <= len(sub_queries) <= MAX_SUB_QUERIES


def merge_documents(results: list[SubQueryResult]):
    merged = []
    seen = set()
    for result in results:
        for doc in result.documents:
            if doc.node_id in seen:
                continue
            seen.add(doc.node_id)
            merged.append(doc)
    return merged


def merge_citations(results: list[SubQueryResult]) -> list[str]:
    merged = []
    seen = set()
    for result in results:
        for citation in result.citations:
            if citation in seen:
                continue
            seen.add(citation)
            merged.append(citation)
    return merged


def merge_node_ids(results: list[SubQueryResult], field_name: str) -> list[str]:
    merged = []
    seen = set()
    for result in results:
        for node_id in getattr(result, field_name, []) or []:
            if node_id in seen:
                continue
            seen.add(node_id)
            merged.append(node_id)
    return merged


def build_sub_query_context(results: list[SubQueryResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Sub-query {index}]",
                    f"Question: {result.sub_query}",
                    f"Evidence summary: {result.evidence_summary or result.answer}",
                    f"Is sufficient: {result.is_sufficient}",
                    f"Fail reason: {result.fail_reason or ''}",
                    f"Diagnostics: {', '.join(result.diagnostics)}",
                ]
            )
        )
    return "\n\n".join(blocks)


def aggregate_sub_query_results(query: str, sub_results: list[SubQueryResult]) -> AggregateResult:
    prompt = SUB_QUERY_AGGREGATE_PROMPT.format(
        query=query,
        sub_query_context=build_sub_query_context(sub_results),
    )
    try:
        result: AggregateResult = LLMService.invoke(
            llm=chatgpt_llm,
            messages=[HumanMessage(content=prompt)],
            schema=AggregateResult,
        )
        return result
    except Exception:
        fallback_summary = "\n\n".join(
            [
                f"{index}. {item.sub_query}\n{item.evidence_summary or item.answer}"
                for index, item in enumerate(sub_results, start=1)
            ]
        )
        return AggregateResult(
            evidence_summary=fallback_summary,
            is_sufficient=any(item.is_sufficient for item in sub_results),
            fail_reason="insufficient_context",
            reason="aggregation_fallback",
        )


def execute_multi_pass_rag(query: RagContext, user_context: dict) -> RAGResult:
    sub_queries = [item.strip() for item in query.decompose_query if item and item.strip()][:MAX_SUB_QUERIES]
    sub_results: list[SubQueryResult] = []
    successful_results: list[SubQueryResult] = []

    for sub_query in sub_queries:
        sub_context = RagContext(
            query=sub_query,
            rewritten_query="",
            expand_query=[],
            decompose_query=[],
            filters=query.filters,
            retrieval_top_k=query.retrieval_top_k,
            rerank_top_k=query.rerank_top_k,
            use_retrieval=query.use_retrieval,
            use_rerank=query.use_rerank,
        )
        result = rag_service.query(sub_context, user_context)
        sub_result = SubQueryResult(
            sub_query=sub_query,
            answer=result.answer or "",
            evidence_summary=result.evidence_summary or result.answer or "",
            citations=result.citations,
            retrieval_candidate_node_ids=result.retrieval_candidate_node_ids,
            rerank_node_ids=result.rerank_node_ids,
            documents=result.documents,
            is_sufficient=result.is_sufficient,
            fail_reason=result.fail_reason,
            diagnostics=result.diagnostics,
        )
        sub_results.append(sub_result)
        if result.success and (result.documents or result.evidence_summary or result.answer):
            successful_results.append(sub_result)

    if not successful_results:
        return RAGResult(
            success=False,
            tool_name="rag",
            answer="子问题执行后仍未获得足够证据。",
            evidence_summary="子问题执行后仍未获得足够证据。",
            is_sufficient=False,
            fail_reason="no_data",
            retrieval_queries=sub_queries,
            retrieval_candidate_node_ids=[],
            rerank_node_ids=[],
            diagnostics=["decompose_multi_pass_failed", f"sub_query_count={len(sub_queries)}"],
            metadata={"sub_query_results": sub_results},
        )

    aggregate_result = aggregate_sub_query_results(query.query or "", sub_results)

    final_result = RAGResult(
        success=True,
        tool_name="rag",
        answer=aggregate_result.evidence_summary or "",
        evidence_summary=aggregate_result.evidence_summary or "",
        documents=merge_documents(successful_results),
        citations=merge_citations(successful_results),
        is_sufficient=aggregate_result.is_sufficient,
        fail_reason=None if aggregate_result.is_sufficient else (aggregate_result.fail_reason or "insufficient_context"),
        reason=aggregate_result.reason,
        retrieval_queries=sub_queries,
        retrieval_candidate_node_ids=merge_node_ids(successful_results, "retrieval_candidate_node_ids"),
        rerank_node_ids=merge_node_ids(successful_results, "rerank_node_ids"),
        diagnostics=[
            "decompose_multi_pass_executed",
            f"sub_query_count={len(sub_queries)}",
            f"sub_query_success_count={len(successful_results)}",
            "sub_query_aggregation_completed" if aggregate_result.is_sufficient else "sub_query_aggregation_finished",
        ],
        metadata={"sub_query_results": sub_results},
    )
    return final_result


def rag_tool(query: RagContext, user_context: dict, previous_result: RAGResult | None = None) -> RAGResult:
    result = (
        execute_multi_pass_rag(query, user_context)
        if should_run_multi_pass(query)
        else rag_service.query(query, user_context, previous_result=previous_result)
    )
    result.tool_name = "rag"

    if result.confidence is None:
        result.confidence = compute_confidence(result)

    if result.fail_reason is None:
        result.fail_reason = decide_fail_reason(result)

    if not result.diagnostics:
        result.diagnostics = ["rag_query_completed"]

    if not result.message:
        result.message = "rag evidence ready" if result.success else "rag evidence failed"
    return result
