#在 top-K 中，命中了多少 ground truth
def recall_at_k(docs, ground_truth):
    top_k_ids = [d["node_id"] for d in docs]
    hit = sum(1 for gt in ground_truth if gt in top_k_ids)
    return hit / len(ground_truth)


# 越早出现越好
def mrr_multi(docs, ground_truth):
    ranks = []
    for gt in ground_truth:
        for i, d in enumerate(docs):
            if d["node_id"] == gt:
                ranks.append(1 / (i + 1))
                break
    return sum(ranks) / len(ground_truth)

#是否“所有必要chunk”都被召回
def coverage(docs, ground_truth):
    top_k_ids = [d["node_id"] for d in docs]
    return int(all(gt in top_k_ids for gt in ground_truth))