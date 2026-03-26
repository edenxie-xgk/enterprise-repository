from core.settings import settings
from tqdm import tqdm


def evaluate_rerank(retrieval,reranker, benchmark, top_k=settings.reranker_top_k):
    top1 = 0

    for item in tqdm(benchmark,desc="重排评估中..."):
        retrieval_data = retrieval.run(item["question"])
        docs = reranker.run(item["question"],docs= retrieval_data,top_k=top_k)
        if not docs:
            continue
        top_doc = docs[0]["node_id"]
        if top_doc in item["node_ids"]:
            top1 += 1

    return top1 / len(benchmark)