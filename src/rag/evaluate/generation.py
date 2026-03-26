from src.rag.evaluate.llm_evaluate_answer import evaluate_answer
from src.rag.generation.generator import generate_answer


def evaluate_generation(llm_service, embed_model, benchmark, retriever):

    correct = 0
    total_score = 0

    for item in benchmark:
        docs = retriever.run([item["question"]], top_k=5)
        answer = generate_answer(llm_service, item["question"], docs)

        # 🔥 LLM评估
        score = evaluate_answer(
            llm_service,
            item["question"],
            item["node_ids"],
            answer["answer"]
        )
        if score > 0.8:
            correct += 1
        total_score += score

    return {
        "accuracy": correct / len(benchmark),
        "avg_score": total_score / len(benchmark)
    }