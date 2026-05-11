from core.settings import settings
from tqdm import tqdm

from src.rag.evaluate.function import coverage, mrr_multi, recall_at_k


def evaluate_rerank(benchmark_cases):
    total_recall = 0.0
    total_mrr = 0.0
    total_coverage = 0

    for case in tqdm(benchmark_cases, desc="evaluate rerank"):
        docs = case.get("rerank_docs") or []
        gt = list(case.get("ground_truth_node_ids") or [])
        if not gt:
            continue

        total_recall += recall_at_k(docs, gt) if docs else 0.0
        total_mrr += mrr_multi(docs, gt) if docs else 0.0
        total_coverage += coverage(docs, gt) if docs else 0

    n = len(benchmark_cases) or 1
    return {
        "recall@k": total_recall / n,
        "mrr": total_mrr / n,
        "coverage": total_coverage / n,
    }


def _docs_from_node_ids(node_ids):
    seen = set()
    docs = []
    for node_id in node_ids or []:
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        docs.append({"node_id": node_id})
    return docs


def _metrics_from_node_ids(node_ids, ground_truth):
    docs = _docs_from_node_ids(node_ids)
    if not ground_truth:
        return {
            "recall@k": 0.0,
            "mrr": 0.0,
            "coverage": 0.0,
        }
    return {
        "recall@k": recall_at_k(docs, ground_truth) if docs else 0.0,
        "mrr": mrr_multi(docs, ground_truth) if docs else 0.0,
        "coverage": coverage(docs, ground_truth) if docs else 0.0,
    }


def _evaluate_node_id_view(benchmark_cases, key: str):
    total_recall = 0.0
    total_mrr = 0.0
    total_coverage = 0.0

    for case in tqdm(benchmark_cases, desc=f"evaluate {key}"):
        gt = list(case.get("ground_truth_node_ids") or [])
        if not gt:
            continue
        metrics = _metrics_from_node_ids(case.get(key) or [], gt)
        total_recall += metrics["recall@k"]
        total_mrr += metrics["mrr"]
        total_coverage += metrics["coverage"]

    n = len(benchmark_cases) or 1
    return {
        "recall@k": total_recall / n,
        "mrr": total_mrr / n,
        "coverage": total_coverage / n,
    }


def build_rerank_diagnostic_flags(case):
    gt = list(case.get("ground_truth_node_ids") or [])
    if not gt:
        return {
            "ranking_regression": False,
            "top_k_truncation_loss": False,
            "score_filter_loss": False,
            "fallback_recovery": False,
            "final_still_worse_than_retrieval": False,
        }

    retrieval_metrics = _metrics_from_node_ids(case.get("retrieval_node_ids") or [], gt)
    full_rank_metrics = _metrics_from_node_ids(case.get("rerank_full_node_ids") or [], gt)
    top_k_only_metrics = _metrics_from_node_ids(case.get("rerank_top_k_only_node_ids") or [], gt)
    threshold_cut_metrics = _metrics_from_node_ids(case.get("rerank_threshold_node_ids") or [], gt)
    final_metrics = _metrics_from_node_ids(case.get("rerank_node_ids") or [], gt)
    eps = 1e-12

    return {
        "ranking_regression": full_rank_metrics["mrr"] + eps < retrieval_metrics["mrr"],
        "top_k_truncation_loss": (
            top_k_only_metrics["coverage"] + eps < full_rank_metrics["coverage"]
            or top_k_only_metrics["mrr"] + eps < full_rank_metrics["mrr"]
        ),
        "score_filter_loss": (
            threshold_cut_metrics["coverage"] + eps < top_k_only_metrics["coverage"]
            or threshold_cut_metrics["mrr"] + eps < top_k_only_metrics["mrr"]
        ),
        "fallback_recovery": (
            final_metrics["coverage"] > threshold_cut_metrics["coverage"] + eps
            or final_metrics["mrr"] > threshold_cut_metrics["mrr"] + eps
        ),
        "final_still_worse_than_retrieval": final_metrics["mrr"] + eps < retrieval_metrics["mrr"],
    }


def evaluate_rerank_diagnostics(benchmark_cases):
    reports = {
        "retrieval_baseline": _evaluate_node_id_view(benchmark_cases, "retrieval_node_ids"),
        "full_rank": _evaluate_node_id_view(benchmark_cases, "rerank_full_node_ids"),
        "top_k_only": _evaluate_node_id_view(benchmark_cases, "rerank_top_k_only_node_ids"),
        "threshold_cut": _evaluate_node_id_view(benchmark_cases, "rerank_threshold_node_ids"),
        "final": _evaluate_node_id_view(benchmark_cases, "rerank_node_ids"),
    }

    ranking_regression_samples = []
    top_k_truncation_loss_samples = []
    score_filter_loss_samples = []
    fallback_recovery_samples = []
    final_still_worse_than_retrieval_samples = []

    for case in benchmark_cases:
        if not list(case.get("ground_truth_node_ids") or []):
            continue

        sample_index = case.get("sample_index")
        flags = build_rerank_diagnostic_flags(case)

        if flags["ranking_regression"]:
            ranking_regression_samples.append(sample_index)

        if flags["top_k_truncation_loss"]:
            top_k_truncation_loss_samples.append(sample_index)

        if flags["score_filter_loss"]:
            score_filter_loss_samples.append(sample_index)

        if flags["fallback_recovery"]:
            fallback_recovery_samples.append(sample_index)

        if flags["final_still_worse_than_retrieval"]:
            final_still_worse_than_retrieval_samples.append(sample_index)

    return {
        "reranker_top_k": settings.reranker_top_k,
        "reranker_min_score": settings.reranker_min_score,
        "reports": reports,
        "sample_counts": {
            "ranking_regression_count": len(ranking_regression_samples),
            "top_k_truncation_loss_count": len(top_k_truncation_loss_samples),
            "score_filter_loss_count": len(score_filter_loss_samples),
            "fallback_recovery_count": len(fallback_recovery_samples),
            "final_still_worse_than_retrieval_count": len(final_still_worse_than_retrieval_samples),
        },
        "sample_indices": {
            "ranking_regression": ranking_regression_samples,
            "top_k_truncation_loss": top_k_truncation_loss_samples,
            "score_filter_loss": score_filter_loss_samples,
            "fallback_recovery": fallback_recovery_samples,
            "final_still_worse_than_retrieval": final_still_worse_than_retrieval_samples,
        },
    }
