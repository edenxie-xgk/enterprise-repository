def evaluate_retrieval(retriever, benchmark, top_k=20):

    recalls = []
    mrrs = []

    for item in benchmark:
        docs = retriever.run([item["query"]], top_k=top_k)
        doc_ids = [d["node_id"] for d in docs]

        # Recall
        recall = int(any(gt in doc_ids for gt in item["ground_truth"]))
        recalls.append(recall)

        # MRR
        rank = next((i for i, doc_id in enumerate(doc_ids) if doc_id in item["ground_truth"]), None)
        mrrs.append(1 / (rank + 1) if rank is not None else 0)

    return {
        "recall": sum(recalls)/len(recalls),
        "mrr": sum(mrrs)/len(mrrs)
    }