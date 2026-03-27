from src.rag.evaluate.llm_evaluate_answer import evaluate_answer
from src.rag.generation.generator import generate_answer


def evaluate_generation(llm,  benchmark, retriever,rerank):
    total_score = 0
    total_correct = 0
    total_citation_correct = 0
    total_retrieval_coverage = 0
    total_rerank_coverage = 0
    total_retrieval_recall = 0
    total_rerank_recall = 0
    for item in benchmark:

        #  retrieval
        retrieval_data = retriever.run([item["question"]])

        # rerank
        rerank_data = rerank.run(item["question"],docs=retrieval_data)

        if not rerank_data:
            continue

        retrieval_ids = [d["node_id"] for d in retrieval_data]
        rerank_ids = [d["node_id"] for d in rerank_data]

        # generation
        answer = generate_answer(llm, item["question"], rerank_data)

        #  LLM评估答案
        score = evaluate_answer(
            llm,
            item["question"],
            item["answer"],
            answer["answer"]
        )

        total_score += score

        if score > 0.8:
            total_correct += 1

        gt_ids = item["node_ids"]

        # Citation Coverage（核心）
        retrieval_coverage = int(all(gt in retrieval_ids for gt in gt_ids))
        total_retrieval_coverage += retrieval_coverage

        rerank_coverage = int(all(gt in rerank_ids for gt in gt_ids))
        total_rerank_coverage += rerank_coverage

        # retrieval recall
        total_retrieval_recall += sum(1 for gt in gt_ids if gt in retrieval_ids) / len(gt_ids)

        # rerank recall
        total_rerank_recall += sum(1 for gt in gt_ids if gt in rerank_ids) / len(gt_ids)

        #  Citation Correctness（引用是否命中）
        if "citations" in answer:
            citation_coverage = int(all(gt in answer["citations"] for gt in gt_ids))
            total_citation_correct += citation_coverage

    n = len(benchmark)

    return {
        #LLM有没有理解对
        "answer_accuracy": total_correct / n,
        "avg_score": total_score / n,
        #是否引用了正确来源（防幻觉）
        "citation_accuracy": total_citation_correct / n,
        # retrieval是否提供足够信息
        "retrieval_coverage": total_retrieval_coverage / n,

        "retrieval_recall": total_retrieval_recall / n,
        "rerank_recall": total_rerank_recall / n,
        #rerank是否提供足够信息
        "rerank_coverage": total_rerank_coverage / n
    }