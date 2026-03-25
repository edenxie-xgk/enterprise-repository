def evaluate_rerank(reranker, benchmark, top_k=5):
    top1 = 0

    for item in benchmark:
        docs = reranker.rerank(item["query"])
        top_doc = docs[0]["metadata"]["chunk_id"]
        if top_doc in item["ground_truth"]:
            top1 += 1

    return top1 / len(benchmark)