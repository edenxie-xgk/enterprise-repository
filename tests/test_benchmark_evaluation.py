import unittest
import importlib.util
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from src.rag.evaluate.generation import evaluate_generation
from src.rag.evaluate.rerank import evaluate_rerank, evaluate_rerank_diagnostics
from src.rag.evaluate.retrieval import evaluate_retrieval
from src.rag.evaluate.text_metrics import exact_match_score, f1_score


RUN_BENCHMARK_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
RUN_BENCHMARK_SPEC = importlib.util.spec_from_file_location("run_benchmark_script", RUN_BENCHMARK_SCRIPT)
RUN_BENCHMARK_MODULE = importlib.util.module_from_spec(RUN_BENCHMARK_SPEC)
RUN_BENCHMARK_SPEC.loader.exec_module(RUN_BENCHMARK_MODULE)


def make_case(
    *,
    sample_index=1,
    qa_id="qa-1",
    question="What are alpha and beta?",
    reference_answer="Alpha is 100 and beta is 10.",
    ground_truth_node_ids=None,
    retrieval_docs=None,
    retrieval_node_ids=None,
    rerank_docs=None,
    rerank_node_ids=None,
    rerank_full_node_ids=None,
    rerank_top_k_only_node_ids=None,
    rerank_threshold_node_ids=None,
    rerank_threshold=0.8,
    search_queries=None,
    retrieval_diagnostics=None,
    rerank_diagnostics=None,
    skipped_reason=None,
):
    gt_ids = list(ground_truth_node_ids or ["node-1", "node-2"])
    retrieval_docs = list(
        retrieval_docs
        or [
            {"node_id": "node-1", "content": "alpha", "rerank_score": 0.95},
            {"node_id": "node-2", "content": "beta", "rerank_score": 0.90},
        ]
    )
    rerank_docs = list(rerank_docs or retrieval_docs)
    return {
        "sample_index": sample_index,
        "qa_id": qa_id,
        "question": question,
        "reference_answer": reference_answer,
        "ground_truth_node_ids": gt_ids,
        "search_queries": list(search_queries or [question]),
        "retrieval_docs": retrieval_docs,
        "retrieval_node_ids": list(retrieval_node_ids or [doc["node_id"] for doc in retrieval_docs]),
        "rerank_docs": rerank_docs,
        "rerank_node_ids": list(rerank_node_ids or [doc["node_id"] for doc in rerank_docs]),
        "rerank_full_node_ids": list(rerank_full_node_ids or [doc["node_id"] for doc in rerank_docs]),
        "rerank_top_k_only_node_ids": list(
            [doc["node_id"] for doc in rerank_docs]
            if rerank_top_k_only_node_ids is None
            else rerank_top_k_only_node_ids
        ),
        "rerank_threshold_node_ids": list(
            [doc["node_id"] for doc in rerank_docs]
            if rerank_threshold_node_ids is None
            else rerank_threshold_node_ids
        ),
        "rerank_threshold": rerank_threshold,
        "retrieval_diagnostics": list(retrieval_diagnostics or ["hybrid_retrieval_executed"]),
        "rerank_diagnostics": list(rerank_diagnostics or ["reranker_executed"]),
        "skipped_reason": skipped_reason,
    }


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_run_benchmark_parse_states(self):
        self.assertEqual(RUN_BENCHMARK_MODULE.parse_states("0,2"), [0, 2])
        self.assertEqual(RUN_BENCHMARK_MODULE.parse_states(" 2,0,2 "), [2, 0])
        self.assertEqual(RUN_BENCHMARK_MODULE.parse_states(""), [0])

    def test_evaluate_retrieval_uses_prepared_benchmark_cases(self):
        benchmark_cases = [make_case()]

        report = evaluate_retrieval(benchmark_cases)

        self.assertEqual(report["recall@k"], 1.0)
        self.assertEqual(report["mrr"], 0.75)
        self.assertEqual(report["coverage"], 1.0)

    def test_evaluate_rerank_uses_prepared_benchmark_cases(self):
        benchmark_cases = [make_case()]

        report = evaluate_rerank(benchmark_cases)

        self.assertEqual(report["recall@k"], 1.0)
        self.assertEqual(report["mrr"], 0.75)
        self.assertEqual(report["coverage"], 1.0)

    def test_evaluate_rerank_diagnostics_separates_ranking_topk_and_threshold_losses(self):
        benchmark_cases = [
            make_case(
                sample_index=1,
                retrieval_node_ids=["node-1", "node-2", "node-3"],
                rerank_full_node_ids=["node-3", "node-1", "node-2"],
                rerank_top_k_only_node_ids=["node-1"],
                rerank_threshold_node_ids=[],
                rerank_node_ids=["node-3", "node-1"],
                rerank_threshold=0.8,
            )
        ]

        report = evaluate_rerank_diagnostics(benchmark_cases)

        self.assertEqual(report["reports"]["retrieval_baseline"]["mrr"], 0.75)
        self.assertEqual(report["reports"]["full_rank"]["mrr"], 0.41666666666666663)
        self.assertEqual(report["reports"]["top_k_only"]["coverage"], 0.0)
        self.assertEqual(report["reports"]["threshold_cut"]["recall@k"], 0.0)
        self.assertEqual(report["reports"]["final"]["coverage"], 0.0)
        self.assertEqual(report["sample_counts"]["ranking_regression_count"], 1)
        self.assertEqual(report["sample_counts"]["top_k_truncation_loss_count"], 1)
        self.assertEqual(report["sample_counts"]["score_filter_loss_count"], 1)
        self.assertEqual(report["sample_counts"]["fallback_recovery_count"], 1)
        self.assertEqual(report["sample_counts"]["final_still_worse_than_retrieval_count"], 1)
        self.assertEqual(report["sample_indices"]["ranking_regression"], [1])
        self.assertEqual(report["sample_indices"]["top_k_truncation_loss"], [1])
        self.assertEqual(report["sample_indices"]["score_filter_loss"], [1])
        self.assertEqual(report["sample_indices"]["fallback_recovery"], [1])

    def test_evaluate_generation_uses_evidence_summary_and_separate_judge_llm(self):
        benchmark_cases = [make_case()]
        generated = SimpleNamespace(
            evidence_summary="Alpha is 100 and beta is 10.",
            citations=["node-1", "node-2"],
        )

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=generated) as evidence_mock,
            patch("src.rag.evaluate.generation.evaluate_answer", return_value=0.92) as answer_mock,
            patch("src.rag.evaluate.generation.evaluate_faithfulness", return_value=0.96) as faithfulness_mock,
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark_cases=benchmark_cases,
            )

        self.assertEqual(evidence_mock.call_args.args[0], "answer-llm")
        self.assertEqual(evidence_mock.call_args.args[1], "What are alpha and beta?")
        self.assertIn("[node_id:node-1]", evidence_mock.call_args.args[2])
        self.assertIn("[node_id:node-2]", evidence_mock.call_args.args[2])
        self.assertEqual(answer_mock.call_args.args[0], "judge-llm")
        self.assertEqual(answer_mock.call_args.args[3], "Alpha is 100 and beta is 10.")
        self.assertEqual(faithfulness_mock.call_args.args[0], "judge-llm")
        self.assertEqual(faithfulness_mock.call_args.args[1], "What are alpha and beta?")
        self.assertIn("[node_id:node-1]", faithfulness_mock.call_args.args[2])
        self.assertEqual(faithfulness_mock.call_args.args[3], "Alpha is 100 and beta is 10.")
        self.assertEqual(report["answer_accuracy"], 1.0)
        self.assertEqual(report["avg_score"], 0.92)
        self.assertEqual(report["faithfulness_accuracy"], 1.0)
        self.assertEqual(report["avg_faithfulness_score"], 0.96)
        self.assertEqual(report["exact_match_accuracy"], 1.0)
        self.assertEqual(report["avg_f1_score"], 1.0)
        self.assertEqual(report["citation_accuracy"], 1.0)
        self.assertEqual(report["retrieval_coverage"], 1.0)
        self.assertEqual(report["retrieval_recall"], 1.0)
        self.assertEqual(report["rerank_recall"], 1.0)
        self.assertEqual(report["rerank_coverage"], 1.0)
        self.assertEqual(report["generation_workers"], 1)

    def test_evaluate_generation_supports_parallel_workers(self):
        benchmark_cases = [
            make_case(sample_index=1, qa_id="qa-1", question="What are alpha and beta?"),
            make_case(sample_index=2, qa_id="qa-2", question="Summarize alpha and beta again."),
        ]
        generated = SimpleNamespace(
            evidence_summary="Alpha is 100 and beta is 10.",
            citations=["node-1", "node-2"],
        )

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=generated),
            patch("src.rag.evaluate.generation.evaluate_answer", return_value=0.92),
            patch("src.rag.evaluate.generation.evaluate_faithfulness", return_value=0.96),
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark_cases=benchmark_cases,
                max_workers=2,
            )

        self.assertEqual(report["generation_workers"], 2)
        self.assertEqual(report["answer_accuracy"], 1.0)
        self.assertEqual(report["avg_score"], 0.92)
        self.assertEqual(report["faithfulness_accuracy"], 1.0)
        self.assertEqual(report["avg_faithfulness_score"], 0.96)
        self.assertEqual(report["exact_match_accuracy"], 1.0)
        self.assertEqual(report["avg_f1_score"], 1.0)
        self.assertEqual(report["citation_accuracy"], 1.0)

    def test_evaluate_generation_tolerates_empty_evidence_payload(self):
        benchmark_cases = [make_case()]

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=None),
            patch("src.rag.evaluate.generation.evaluate_answer") as answer_mock,
            patch("src.rag.evaluate.generation.evaluate_faithfulness") as faithfulness_mock,
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark_cases=benchmark_cases,
            )

        answer_mock.assert_not_called()
        faithfulness_mock.assert_not_called()
        self.assertEqual(report["answer_accuracy"], 0.0)
        self.assertEqual(report["avg_score"], 0.0)
        self.assertEqual(report["faithfulness_accuracy"], 0.0)
        self.assertEqual(report["avg_faithfulness_score"], 0.0)
        self.assertEqual(report["exact_match_accuracy"], 0.0)
        self.assertEqual(report["avg_f1_score"], 0.0)
        self.assertEqual(report["citation_accuracy"], 0.0)
        self.assertEqual(report["retrieval_coverage"], 1.0)
        self.assertEqual(report["retrieval_recall"], 1.0)
        self.assertEqual(report["rerank_recall"], 1.0)
        self.assertEqual(report["rerank_coverage"], 1.0)
        self.assertEqual(report["generation_workers"], 1)
        self.assertEqual(report["evidence_generation_failed"], 0)
        self.assertEqual(report["answer_evaluation_failed"], 0)
        self.assertEqual(report["faithfulness_evaluation_failed"], 0)

    def test_evaluate_generation_returns_sample_details(self):
        benchmark_cases = [
            make_case(
                sample_index=1,
                qa_id="qa-1",
                question="What is the summary?",
                reference_answer="Alpha and beta.",
                search_queries=["What is the summary?"],
            )
        ]
        generated = SimpleNamespace(
            evidence_summary="Alpha and beta.",
            citations=["node-1", "node-2"],
        )

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=generated),
            patch("src.rag.evaluate.generation.evaluate_answer", return_value=0.92),
            patch("src.rag.evaluate.generation.evaluate_faithfulness", return_value=0.96),
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark_cases=benchmark_cases,
                include_details=True,
            )

        self.assertIn("sample_details", report)
        self.assertEqual(len(report["sample_details"]), 1)
        detail = report["sample_details"][0]
        self.assertEqual(detail["sample_index"], 1)
        self.assertEqual(detail["qa_id"], "qa-1")
        self.assertEqual(detail["question"], "What is the summary?")
        self.assertEqual(detail["reference_answer"], "Alpha and beta.")
        self.assertEqual(detail["ground_truth_node_ids"], ["node-1", "node-2"])
        self.assertEqual(detail["search_queries"], ["What is the summary?"])
        self.assertEqual(detail["retrieval_node_ids"], ["node-1", "node-2"])
        self.assertEqual(detail["retrieval_candidate_count"], 2)
        self.assertEqual(detail["retrieval_hit_count"], 2)
        self.assertEqual(detail["retrieval_recall"], 1.0)
        self.assertEqual(detail["retrieval_mrr"], 0.75)
        self.assertEqual(detail["retrieval_coverage"], 1)
        self.assertEqual(detail["rerank_node_ids"], ["node-1", "node-2"])
        self.assertEqual(detail["rerank_full_count"], 2)
        self.assertEqual(detail["rerank_top_k_only_count"], 2)
        self.assertEqual(detail["rerank_threshold_count"], 2)
        self.assertEqual(detail["rerank_final_count"], 2)
        self.assertEqual(detail["rerank_hit_count"], 2)
        self.assertEqual(detail["rerank_recall"], 1.0)
        self.assertEqual(detail["rerank_mrr"], 0.75)
        self.assertEqual(detail["rerank_coverage"], 1)
        self.assertEqual(detail["retrieval_diagnostics"], ["hybrid_retrieval_executed"])
        self.assertEqual(detail["rerank_diagnostics"], ["reranker_executed"])
        self.assertEqual(
            detail["rerank_diagnostic_flags"],
            {
                "ranking_regression": False,
                "top_k_truncation_loss": False,
                "score_filter_loss": False,
                "fallback_recovery": False,
                "final_still_worse_than_retrieval": False,
            },
        )
        self.assertEqual(detail["generated_answer"], "Alpha and beta.")
        self.assertEqual(detail["answer_score"], 0.92)
        self.assertTrue(detail["answer_passed"])
        self.assertEqual(detail["faithfulness_score"], 0.96)
        self.assertTrue(detail["faithfulness_passed"])
        self.assertEqual(detail["exact_match"], 1.0)
        self.assertEqual(detail["f1_score"], 1.0)
        self.assertEqual(detail["answer_citations"], ["node-1", "node-2"])
        self.assertEqual(detail["citation_hit_count"], 2)
        self.assertEqual(detail["citation_coverage"], 1)
        self.assertEqual(detail["faithfulness_evaluation_failed"], 0)
        self.assertIsNone(detail["skipped_reason"])

    def test_evaluate_generation_returns_rerank_diagnostic_flags_in_sample_details(self):
        benchmark_cases = [
            make_case(
                sample_index=1,
                qa_id="qa-1",
                question="What is the summary?",
                reference_answer="Alpha and beta.",
                retrieval_node_ids=["node-1", "node-2", "node-3"],
                rerank_full_node_ids=["node-3", "node-1", "node-2"],
                rerank_top_k_only_node_ids=["node-1"],
                rerank_threshold_node_ids=[],
                rerank_node_ids=["node-3", "node-1"],
                rerank_threshold=0.8,
            )
        ]
        generated = SimpleNamespace(
            evidence_summary="Alpha and beta.",
            citations=["node-1", "node-2"],
        )

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=generated),
            patch("src.rag.evaluate.generation.evaluate_answer", return_value=0.92),
            patch("src.rag.evaluate.generation.evaluate_faithfulness", return_value=0.96),
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark_cases=benchmark_cases,
                include_details=True,
            )

        detail = report["sample_details"][0]
        self.assertEqual(detail["rerank_full_count"], 3)
        self.assertEqual(detail["rerank_top_k_only_count"], 1)
        self.assertEqual(detail["rerank_threshold_count"], 0)
        self.assertEqual(detail["rerank_final_count"], 2)
        self.assertEqual(
            detail["rerank_diagnostic_flags"],
            {
                "ranking_regression": True,
                "top_k_truncation_loss": True,
                "score_filter_loss": True,
                "fallback_recovery": True,
                "final_still_worse_than_retrieval": True,
            },
        )

    def test_text_metrics_normalize_spacing_and_punctuation(self):
        self.assertEqual(exact_match_score("15.4%", "15.4 %"), 1.0)
        self.assertEqual(exact_match_score("中国人寿，2024年。", "中国人寿2024年"), 1.0)
        self.assertAlmostEqual(f1_score("2024年同比增长7.7%", "同比增长7.7%"), 0.7619047619047619)


if __name__ == "__main__":
    unittest.main()
