from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.rag.context.builder import ContextBuilder
from src.rag.evaluate.function import mrr_multi
from src.rag.evaluate.llm_evaluate_answer import evaluate_answer
from src.rag.evaluate.llm_evaluate_faithfulness import evaluate_faithfulness
from src.rag.evaluate.rerank import build_rerank_diagnostic_flags
from src.rag.evaluate.text_metrics import exact_match_score, f1_score
from src.rag.generation.generator import evaluate_evidence
from utils.logger_handler import logger


def _prepare_generation_item(case, context_builder):
    question = case.get("question") or ""
    gt_ids = list(case.get("ground_truth_node_ids") or [])
    retrieval_docs = list(case.get("retrieval_docs") or [])
    rerank_docs = list(case.get("rerank_docs") or [])
    retrieval_ids = list(case.get("retrieval_node_ids") or [])
    rerank_ids = list(case.get("rerank_node_ids") or [])
    rerank_full_ids = list(case.get("rerank_full_node_ids") or [])
    rerank_top_k_only_ids = list(case.get("rerank_top_k_only_node_ids") or [])
    rerank_threshold_ids = list(case.get("rerank_threshold_node_ids") or [])
    diagnostic_flags = build_rerank_diagnostic_flags(case)

    skipped_reason = case.get("skipped_reason")
    if not skipped_reason and not question:
        skipped_reason = "missing_question"
    if not skipped_reason and not gt_ids:
        skipped_reason = "missing_node_ids"

    retrieval_hit_count = sum(1 for gt in gt_ids if gt in retrieval_ids)
    rerank_hit_count = sum(1 for gt in gt_ids if gt in rerank_ids)

    return {
        "sample_index": case.get("sample_index"),
        "qa_id": case.get("qa_id"),
        "question": question,
        "reference_answer": case.get("reference_answer") or "",
        "ground_truth_node_ids": gt_ids,
        "search_queries": list(case.get("search_queries") or []),
        "retrieval_ids": retrieval_ids,
        "retrieval_hit_count": retrieval_hit_count,
        "retrieval_recall": retrieval_hit_count / len(gt_ids) if gt_ids else 0.0,
        "retrieval_mrr": mrr_multi(retrieval_docs, gt_ids) if retrieval_docs and gt_ids else 0.0,
        "retrieval_coverage": int(all(gt in retrieval_ids for gt in gt_ids)) if gt_ids else 0,
        "rerank_ids": rerank_ids,
        "rerank_full_count": len(rerank_full_ids),
        "rerank_top_k_only_count": len(rerank_top_k_only_ids),
        "rerank_threshold_count": len(rerank_threshold_ids),
        "rerank_final_count": len(rerank_ids),
        "rerank_full_ids": rerank_full_ids,
        "rerank_top_k_only_ids": rerank_top_k_only_ids,
        "rerank_threshold_ids": rerank_threshold_ids,
        "rerank_threshold": case.get("rerank_threshold"),
        "rerank_hit_count": rerank_hit_count,
        "rerank_recall": rerank_hit_count / len(gt_ids) if gt_ids else 0.0,
        "rerank_mrr": mrr_multi(rerank_docs, gt_ids) if rerank_docs and gt_ids else 0.0,
        "rerank_coverage": int(all(gt in rerank_ids for gt in gt_ids)) if gt_ids else 0,
        "context": context_builder.run(rerank_docs) if rerank_docs else "",
        "has_rerank_data": bool(rerank_docs),
        "retrieval_diagnostics": list(case.get("retrieval_diagnostics") or []),
        "rerank_diagnostics": list(case.get("rerank_diagnostics") or []),
        "rerank_diagnostic_flags": diagnostic_flags,
        "skipped_reason": skipped_reason,
    }


def _score_generation_item(prepared_item, answer_llm, judge_llm):
    question = prepared_item["question"]
    context = prepared_item["context"]
    gt_ids = prepared_item["ground_truth_node_ids"]
    reference_answer = prepared_item["reference_answer"]
    answer_text = ""
    answer_citations = []
    evidence_generation_failed = 0
    answer_evaluation_failed = 0
    faithfulness_evaluation_failed = 0

    if prepared_item["skipped_reason"]:
        return {
            "generated_answer": "",
            "answer_score": 0.0,
            "answer_passed": False,
            "faithfulness_score": 0.0,
            "faithfulness_passed": False,
            "exact_match": 0.0,
            "f1_score": 0.0,
            "answer_citations": [],
            "citation_hit_count": 0,
            "citation_coverage": 0,
            "evidence_generation_failed": 0,
            "answer_evaluation_failed": 0,
            "faithfulness_evaluation_failed": 0,
        }

    if prepared_item["has_rerank_data"] and prepared_item["context"]:
        try:
            response = evaluate_evidence(
                answer_llm,
                question,
                prepared_item["context"],
                min_citation_count=min(2, len(gt_ids)) if gt_ids else 1,
            )
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
                reference_answer,
                answer_text,
            )
        except Exception as exc:
            logger.warning(f"[benchmark] answer evaluation failed for question={question!r}: {exc}")
            score = 0.0
            answer_evaluation_failed += 1

        try:
            faithfulness_score = evaluate_faithfulness(
                judge_llm,
                question,
                context,
                answer_text,
            )
        except Exception as exc:
            logger.warning(f"[benchmark] faithfulness evaluation failed for question={question!r}: {exc}")
            faithfulness_score = 0.0
            faithfulness_evaluation_failed += 1
    else:
        score = 0.0
        faithfulness_score = 0.0

    citation_hit_count = sum(1 for gt in gt_ids if gt in answer_citations)
    citation_coverage = int(all(gt in answer_citations for gt in gt_ids)) if gt_ids else 0
    return {
        "generated_answer": answer_text,
        "answer_score": score,
        "answer_passed": score >= 0.8,
        "faithfulness_score": faithfulness_score,
        "faithfulness_passed": faithfulness_score >= 0.8,
        "exact_match": exact_match_score(answer_text, reference_answer),
        "f1_score": f1_score(answer_text, reference_answer),
        "answer_citations": answer_citations,
        "citation_hit_count": citation_hit_count,
        "citation_coverage": citation_coverage,
        "evidence_generation_failed": evidence_generation_failed,
        "answer_evaluation_failed": answer_evaluation_failed,
        "faithfulness_evaluation_failed": faithfulness_evaluation_failed,
    }


def evaluate_generation(
    answer_llm,
    benchmark_cases,
    judge_llm=None,
    max_workers: int = 1,
    include_details: bool = False,
):
    total_score = 0.0
    total_correct = 0
    total_faithfulness_score = 0.0
    total_faithfulness_correct = 0
    total_exact_match = 0.0
    total_f1_score = 0.0
    total_citation_correct = 0
    total_retrieval_coverage = 0
    total_rerank_coverage = 0
    total_retrieval_recall = 0.0
    total_rerank_recall = 0.0
    evidence_generation_failed = 0
    answer_evaluation_failed = 0
    faithfulness_evaluation_failed = 0
    judge = judge_llm or answer_llm
    context_builder = ContextBuilder()

    prepared_items = [
        _prepare_generation_item(case, context_builder)
        for case in benchmark_cases
    ]

    for prepared_item in prepared_items:
        total_retrieval_coverage += prepared_item["retrieval_coverage"]
        total_rerank_coverage += prepared_item["rerank_coverage"]
        total_retrieval_recall += prepared_item["retrieval_recall"]
        total_rerank_recall += prepared_item["rerank_recall"]

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
        total_faithfulness_score += result["faithfulness_score"]
        total_exact_match += result["exact_match"]
        total_f1_score += result["f1_score"]
        evidence_generation_failed += result["evidence_generation_failed"]
        answer_evaluation_failed += result["answer_evaluation_failed"]
        faithfulness_evaluation_failed += result["faithfulness_evaluation_failed"]

        if result["answer_passed"]:
            total_correct += 1

        if result["faithfulness_passed"]:
            total_faithfulness_correct += 1

        total_citation_correct += result["citation_coverage"]

        if include_details:
            sample_details.append(
                {
                    "sample_index": prepared_item["sample_index"],
                    "qa_id": prepared_item["qa_id"],
                    "question": prepared_item["question"],
                    "reference_answer": prepared_item["reference_answer"],
                    "ground_truth_node_ids": prepared_item["ground_truth_node_ids"],
                    "search_queries": prepared_item["search_queries"],
                    "retrieval_node_ids": prepared_item["retrieval_ids"],
                    "retrieval_candidate_count": len(prepared_item["retrieval_ids"]),
                    "retrieval_hit_count": prepared_item["retrieval_hit_count"],
                    "retrieval_recall": prepared_item["retrieval_recall"],
                    "retrieval_mrr": prepared_item["retrieval_mrr"],
                    "retrieval_coverage": prepared_item["retrieval_coverage"],
                    "rerank_node_ids": prepared_item["rerank_ids"],
                    "rerank_full_count": prepared_item["rerank_full_count"],
                    "rerank_top_k_only_count": prepared_item["rerank_top_k_only_count"],
                    "rerank_threshold_count": prepared_item["rerank_threshold_count"],
                    "rerank_final_count": prepared_item["rerank_final_count"],
                    "rerank_full_node_ids": prepared_item["rerank_full_ids"],
                    "rerank_top_k_only_node_ids": prepared_item["rerank_top_k_only_ids"],
                    "rerank_threshold_node_ids": prepared_item["rerank_threshold_ids"],
                    "rerank_threshold": prepared_item["rerank_threshold"],
                    "rerank_hit_count": prepared_item["rerank_hit_count"],
                    "rerank_recall": prepared_item["rerank_recall"],
                    "rerank_mrr": prepared_item["rerank_mrr"],
                    "rerank_coverage": prepared_item["rerank_coverage"],
                    "retrieval_diagnostics": prepared_item["retrieval_diagnostics"],
                    "rerank_diagnostics": prepared_item["rerank_diagnostics"],
                    "rerank_diagnostic_flags": prepared_item["rerank_diagnostic_flags"],
                    "generated_answer": result["generated_answer"],
                    "answer_score": result["answer_score"],
                    "answer_passed": result["answer_passed"],
                    "faithfulness_score": result["faithfulness_score"],
                    "faithfulness_passed": result["faithfulness_passed"],
                    "exact_match": result["exact_match"],
                    "f1_score": result["f1_score"],
                    "answer_citations": result["answer_citations"],
                    "citation_hit_count": result["citation_hit_count"],
                    "citation_coverage": result["citation_coverage"],
                    "evidence_generation_failed": result["evidence_generation_failed"],
                    "answer_evaluation_failed": result["answer_evaluation_failed"],
                    "faithfulness_evaluation_failed": result["faithfulness_evaluation_failed"],
                    "skipped_reason": prepared_item["skipped_reason"],
                }
            )

    n = len(benchmark_cases) or 1
    report = {
        "answer_accuracy": total_correct / n,
        "avg_score": total_score / n,
        "faithfulness_accuracy": total_faithfulness_correct / n,
        "avg_faithfulness_score": total_faithfulness_score / n,
        "exact_match_accuracy": total_exact_match / n,
        "avg_f1_score": total_f1_score / n,
        "citation_accuracy": total_citation_correct / n,
        "retrieval_coverage": total_retrieval_coverage / n,
        "retrieval_recall": total_retrieval_recall / n,
        "rerank_recall": total_rerank_recall / n,
        "rerank_coverage": total_rerank_coverage / n,
        "generation_workers": worker_count,
        "evidence_generation_failed": evidence_generation_failed,
        "answer_evaluation_failed": answer_evaluation_failed,
        "faithfulness_evaluation_failed": faithfulness_evaluation_failed,
    }
    if include_details:
        report["sample_details"] = sample_details
    return report
