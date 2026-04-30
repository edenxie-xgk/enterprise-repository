from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.rag.context.builder import ContextBuilder
from src.rag.evaluate.function import mrr_multi
from src.rag.evaluate.llm_evaluate_answer import evaluate_answer
from src.rag.generation.generator import evaluate_evidence
from utils.logger_handler import logger


def _extract_node_ids(docs):
    node_ids = []
    seen = set()
    for doc in docs or []:
        if isinstance(doc, dict):
            node_id = doc.get("node_id")
        else:
            node_id = getattr(doc, "node_id", None)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node_ids.append(node_id)
    return node_ids


def _build_base_detail(item, sample_index):
    return {
        "sample_index": sample_index,
        "qa_id": str(item.get("_id")) if item.get("_id") is not None else None,
        "question": item.get("question") or "",
        "reference_answer": item.get("answer") or "",
        "ground_truth_node_ids": list(item.get("node_ids") or []),
    }


def _prepare_generation_item(item, retriever, rerank, context_builder, sample_index):
    prepared = _build_base_detail(item, sample_index)
    question = prepared["question"]
    gt_ids = prepared["ground_truth_node_ids"]

    prepared.update(
        {
            "retrieval_ids": [],
            "rerank_ids": [],
            "retrieval_hit_count": 0,
            "retrieval_recall": 0.0,
            "retrieval_mrr": 0.0,
            "retrieval_coverage": 0,
            "rerank_hit_count": 0,
            "rerank_recall": 0.0,
            "rerank_mrr": 0.0,
            "rerank_coverage": 0,
            "context": "",
            "has_rerank_data": False,
            "skipped_reason": None,
        }
    )

    if not question:
        prepared["skipped_reason"] = "missing_question"
        return prepared

    if not gt_ids:
        prepared["skipped_reason"] = "missing_node_ids"
        return prepared

    retrieval_data = retriever.run([question]) or []
    retrieval_ids = _extract_node_ids(retrieval_data)
    retrieval_hit_count = sum(1 for gt in gt_ids if gt in retrieval_ids)

    rerank_data = rerank.run(question, docs=retrieval_data) or []
    rerank_ids = _extract_node_ids(rerank_data)
    rerank_hit_count = sum(1 for gt in gt_ids if gt in rerank_ids)

    prepared.update(
        {
            "retrieval_ids": retrieval_ids,
            "rerank_ids": rerank_ids,
            "retrieval_hit_count": retrieval_hit_count,
            "retrieval_recall": retrieval_hit_count / len(gt_ids),
            "retrieval_mrr": mrr_multi(retrieval_data, gt_ids) if retrieval_data else 0.0,
            "retrieval_coverage": int(all(gt in retrieval_ids for gt in gt_ids)),
            "rerank_hit_count": rerank_hit_count,
            "rerank_recall": rerank_hit_count / len(gt_ids),
            "rerank_mrr": mrr_multi(rerank_data, gt_ids) if rerank_data else 0.0,
            "rerank_coverage": int(all(gt in rerank_ids for gt in gt_ids)),
            "context": context_builder.run(rerank_data) if rerank_data else "",
            "has_rerank_data": bool(rerank_data),
        }
    )
    return prepared


def _score_generation_item(prepared_item, answer_llm, judge_llm):
    question = prepared_item["question"]
    gt_ids = prepared_item["ground_truth_node_ids"]
    answer_text = ""
    answer_citations = []
    evidence_generation_failed = 0
    answer_evaluation_failed = 0

    if prepared_item["skipped_reason"]:
        return {
            "generated_answer": "",
            "answer_score": 0.0,
            "answer_passed": False,
            "answer_citations": [],
            "citation_hit_count": 0,
            "citation_coverage": 0,
            "evidence_generation_failed": 0,
            "answer_evaluation_failed": 0,
        }

    if prepared_item["has_rerank_data"] and prepared_item["context"]:
        try:
            response = evaluate_evidence(answer_llm, question, prepared_item["context"])
        except Exception as exc:
            logger.warning(f"[benchmark] evidence generation failed for question={question!r}: {exc}")
            response = None
            evidence_generation_failed += 1

        answer_text = getattr(response, "evidence_summary", "") or ""
        answer_citations = list(getattr(response, "citations", []) or [])

    if answer_text:
        try:
            score = evaluate_answer(
                judge_llm,
                question,
                prepared_item["reference_answer"],
                answer_text,
            )
        except Exception as exc:
            logger.warning(f"[benchmark] answer evaluation failed for question={question!r}: {exc}")
            score = 0.0
            answer_evaluation_failed += 1
    else:
        score = 0.0

    citation_hit_count = sum(1 for gt in gt_ids if gt in answer_citations)
    citation_coverage = int(all(gt in answer_citations for gt in gt_ids)) if gt_ids else 0
    return {
        "generated_answer": answer_text,
        "answer_score": score,
        "answer_passed": score >= 0.8,
        "answer_citations": answer_citations,
        "citation_hit_count": citation_hit_count,
        "citation_coverage": citation_coverage,
        "evidence_generation_failed": evidence_generation_failed,
        "answer_evaluation_failed": answer_evaluation_failed,
    }


def evaluate_generation(
    answer_llm,
    benchmark,
    retriever,
    rerank,
    judge_llm=None,
    max_workers: int = 1,
    include_details: bool = False,
):
    total_score = 0.0
    total_correct = 0
    total_citation_correct = 0
    total_retrieval_coverage = 0
    total_rerank_coverage = 0
    total_retrieval_recall = 0.0
    total_rerank_recall = 0.0
    evidence_generation_failed = 0
    answer_evaluation_failed = 0
    judge = judge_llm or answer_llm
    context_builder = ContextBuilder()
    prepared_items = []

    for sample_index, item in enumerate(benchmark, start=1):
        prepared = _prepare_generation_item(item, retriever, rerank, context_builder, sample_index)
        total_retrieval_coverage += prepared["retrieval_coverage"]
        total_rerank_coverage += prepared["rerank_coverage"]
        total_retrieval_recall += prepared["retrieval_recall"]
        total_rerank_recall += prepared["rerank_recall"]
        prepared_items.append(prepared)

    worker_count = max(1, int(max_workers or 1))
    if worker_count == 1:
        generation_results = [
            _score_generation_item(prepared_item, answer_llm, judge)
            for prepared_item in tqdm(prepared_items, desc="evaluate generation")
        ]
    else:
        generation_results = [None] * len(prepared_items)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_score_generation_item, prepared_item, answer_llm, judge): prepared_item["sample_index"]
                for prepared_item in prepared_items
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="evaluate generation"):
                sample_index = futures[future]
                generation_results[sample_index - 1] = future.result()

    sample_details = []
    for prepared_item, result in zip(prepared_items, generation_results):
        total_score += result["answer_score"]
        evidence_generation_failed += result["evidence_generation_failed"]
        answer_evaluation_failed += result["answer_evaluation_failed"]

        if result["answer_passed"]:
            total_correct += 1

        total_citation_correct += result["citation_coverage"]

        if include_details:
            sample_details.append(
                {
                    "sample_index": prepared_item["sample_index"],
                    "qa_id": prepared_item["qa_id"],
                    "question": prepared_item["question"],
                    "reference_answer": prepared_item["reference_answer"],
                    "ground_truth_node_ids": prepared_item["ground_truth_node_ids"],
                    "retrieval_node_ids": prepared_item["retrieval_ids"],
                    "retrieval_hit_count": prepared_item["retrieval_hit_count"],
                    "retrieval_recall": prepared_item["retrieval_recall"],
                    "retrieval_mrr": prepared_item["retrieval_mrr"],
                    "retrieval_coverage": prepared_item["retrieval_coverage"],
                    "rerank_node_ids": prepared_item["rerank_ids"],
                    "rerank_hit_count": prepared_item["rerank_hit_count"],
                    "rerank_recall": prepared_item["rerank_recall"],
                    "rerank_mrr": prepared_item["rerank_mrr"],
                    "rerank_coverage": prepared_item["rerank_coverage"],
                    "generated_answer": result["generated_answer"],
                    "answer_score": result["answer_score"],
                    "answer_passed": result["answer_passed"],
                    "answer_citations": result["answer_citations"],
                    "citation_hit_count": result["citation_hit_count"],
                    "citation_coverage": result["citation_coverage"],
                    "evidence_generation_failed": result["evidence_generation_failed"],
                    "answer_evaluation_failed": result["answer_evaluation_failed"],
                    "skipped_reason": prepared_item["skipped_reason"],
                }
            )

    n = len(benchmark) or 1
    report = {
        "answer_accuracy": total_correct / n,
        "avg_score": total_score / n,
        "citation_accuracy": total_citation_correct / n,
        "retrieval_coverage": total_retrieval_coverage / n,
        "retrieval_recall": total_retrieval_recall / n,
        "rerank_recall": total_rerank_recall / n,
        "rerank_coverage": total_rerank_coverage / n,
        "generation_workers": worker_count,
        "evidence_generation_failed": evidence_generation_failed,
        "answer_evaluation_failed": answer_evaluation_failed,
    }
    if include_details:
        report["sample_details"] = sample_details
    return report
