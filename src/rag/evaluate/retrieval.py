from tqdm import tqdm

from src.rag.evaluate.function import coverage, mrr_multi, recall_at_k


def evaluate_retrieval(benchmark_cases):
    total_recall = 0.0
    total_mrr = 0.0
    total_coverage = 0

    for case in tqdm(benchmark_cases, desc="evaluate retrieval"):
        docs = case.get("retrieval_docs") or []
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
