from typing import Iterable

from src.rag.rag_service import rag_service
from src.types.rag_type import RAGResult, RagContext


DENSE_SCORE_THRESHOLD = 0.3
RERANK_SCORE_THRESHOLD = 0.5
CONFIDENCE_WEIGHT = {"dense": 0.4, "bm25": 0.3, "rerank": 0.3}


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


def rag_tool(query: RagContext, user_context: dict) -> RAGResult:
    result = rag_service.query(query, user_context)
    result.tool_name = "rag"

    if result.confidence is None:
        result.confidence = compute_confidence(result)

    if result.fail_reason is None:
        result.fail_reason = decide_fail_reason(result)

    if not result.diagnostics:
        result.diagnostics = ["rag_query_completed"]

    if not result.message:
        result.message = "rag query success" if result.success else "rag query failed"

    return result
