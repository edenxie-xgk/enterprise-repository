from tqdm import tqdm

from core.settings import settings
from src.rag.evaluate.function import mrr_multi, recall_at_k, coverage


def evaluate_retrieval(retriever, benchmark):

    total_recall = 0
    total_mrr = 0
    total_coverage = 0

    for item in tqdm(benchmark, desc="召回评估中..."):

        docs = retriever.run([item["question"]])
        if not docs:
            continue

        gt = item["node_ids"]

        # Recall@K（比例）
        recall = recall_at_k(docs, gt)
        total_recall += recall

        # MRR
        mrr = mrr_multi(docs, gt)
        total_mrr += mrr

        # Coverage（是否全部命中）
        cov = coverage(docs, gt)
        total_coverage += cov

    n = len(benchmark)

    return {
        "recall@k": total_recall / n,
        "mrr": total_mrr / n,
        "coverage": total_coverage / n
    }