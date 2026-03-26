from tqdm import tqdm

from core.settings import settings


def evaluate_retrieval(retriever, benchmark, top_k=settings.retriever_top_k):

    recalls = []
    mrrs = []

    for item in tqdm(benchmark,desc="召回评估中..."):
        docs = retriever.run([item["question"]], top_k=top_k)
        doc_ids = [d["node_id"] for d in docs]

        # Recall
        recall = int(any(gt in doc_ids for gt in item["node_ids"]))
        recalls.append(recall)

        # MRR
        rank = next((i for i, doc_id in enumerate(doc_ids) if doc_id in item["node_ids"]), None)
        mrrs.append(1 / (rank + 1) if rank is not None else 0)

    return {
        "recall": sum(recalls)/len(recalls),
        "mrr": sum(mrrs)/len(mrrs)
    }