import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.rag.evaluate.generation import evaluate_generation
from src.rag.evaluate.rerank import evaluate_rerank


class FakeRetriever:
    def __init__(self, docs):
        self.docs = [dict(doc) for doc in docs]
        self.calls = []

    def run(self, search_queries, **kwargs):
        self.calls.append(search_queries)
        return [dict(doc) for doc in self.docs]


class FakeReranker:
    def __init__(self, docs):
        self.docs = [dict(doc) for doc in docs]
        self.calls = []

    def run(self, query, docs, top_k=None, **kwargs):
        self.calls.append({"query": query, "docs": [dict(doc) for doc in docs], "top_k": top_k})
        return [dict(doc) for doc in self.docs]


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_evaluate_rerank_uses_full_question_list_for_retrieval(self):
        docs = [
            {"node_id": "node-1", "content": "alpha", "rerank_score": 0.95},
            {"node_id": "node-2", "content": "beta", "rerank_score": 0.90},
        ]
        retriever = FakeRetriever(docs)
        reranker = FakeReranker(docs)
        benchmark = [
            {
                "question": "What are alpha and beta?",
                "node_ids": ["node-1", "node-2"],
            }
        ]

        report = evaluate_rerank(retriever, reranker, benchmark, top_k=3)

        self.assertEqual(retriever.calls, [["What are alpha and beta?"]])
        self.assertEqual(reranker.calls[0]["query"], "What are alpha and beta?")
        self.assertEqual(report["recall@k"], 1.0)
        self.assertEqual(report["mrr"], 0.75)
        self.assertEqual(report["coverage"], 1.0)

    def test_evaluate_generation_uses_evidence_summary_and_separate_judge_llm(self):
        docs = [
            {"node_id": "node-1", "content": "alpha", "rerank_score": 0.95},
            {"node_id": "node-2", "content": "beta", "rerank_score": 0.90},
        ]
        retriever = FakeRetriever(docs)
        reranker = FakeReranker(docs)
        benchmark = [
            {
                "question": "What are alpha and beta?",
                "answer": "Alpha is 100 and beta is 10.",
                "node_ids": ["node-1", "node-2"],
            }
        ]
        generated = SimpleNamespace(
            evidence_summary="Alpha is 100 and beta is 10.",
            citations=["node-1", "node-2"],
        )

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=generated) as evidence_mock,
            patch("src.rag.evaluate.generation.evaluate_answer", return_value=0.92) as answer_mock,
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark=benchmark,
                retriever=retriever,
                rerank=reranker,
            )

        self.assertEqual(retriever.calls, [["What are alpha and beta?"]])
        self.assertEqual(evidence_mock.call_args.args[0], "answer-llm")
        self.assertEqual(evidence_mock.call_args.args[1], "What are alpha and beta?")
        self.assertIn("[node_id:node-1]", evidence_mock.call_args.args[2])
        self.assertIn("[node_id:node-2]", evidence_mock.call_args.args[2])
        self.assertEqual(answer_mock.call_args.args[0], "judge-llm")
        self.assertEqual(answer_mock.call_args.args[3], "Alpha is 100 and beta is 10.")
        self.assertEqual(report["answer_accuracy"], 1.0)
        self.assertEqual(report["avg_score"], 0.92)
        self.assertEqual(report["citation_accuracy"], 1.0)
        self.assertEqual(report["retrieval_coverage"], 1.0)
        self.assertEqual(report["retrieval_recall"], 1.0)
        self.assertEqual(report["rerank_recall"], 1.0)
        self.assertEqual(report["rerank_coverage"], 1.0)
        self.assertEqual(report["generation_workers"], 1)

    def test_evaluate_generation_supports_parallel_workers(self):
        docs = [
            {"node_id": "node-1", "content": "alpha", "rerank_score": 0.95},
            {"node_id": "node-2", "content": "beta", "rerank_score": 0.90},
        ]
        retriever = FakeRetriever(docs)
        reranker = FakeReranker(docs)
        benchmark = [
            {
                "question": "What are alpha and beta?",
                "answer": "Alpha is 100 and beta is 10.",
                "node_ids": ["node-1", "node-2"],
            },
            {
                "question": "Summarize alpha and beta again.",
                "answer": "Alpha is 100 and beta is 10.",
                "node_ids": ["node-1", "node-2"],
            },
        ]
        generated = SimpleNamespace(
            evidence_summary="Alpha is 100 and beta is 10.",
            citations=["node-1", "node-2"],
        )

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=generated),
            patch("src.rag.evaluate.generation.evaluate_answer", return_value=0.92),
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark=benchmark,
                retriever=retriever,
                rerank=reranker,
                max_workers=2,
            )

        self.assertEqual(report["generation_workers"], 2)
        self.assertEqual(report["answer_accuracy"], 1.0)
        self.assertEqual(report["avg_score"], 0.92)
        self.assertEqual(report["citation_accuracy"], 1.0)

    def test_evaluate_generation_tolerates_empty_evidence_payload(self):
        docs = [
            {"node_id": "node-1", "content": "alpha", "rerank_score": 0.95},
            {"node_id": "node-2", "content": "beta", "rerank_score": 0.90},
        ]
        retriever = FakeRetriever(docs)
        reranker = FakeReranker(docs)
        benchmark = [
            {
                "question": "What are alpha and beta?",
                "answer": "Alpha is 100 and beta is 10.",
                "node_ids": ["node-1", "node-2"],
            }
        ]

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=None),
            patch("src.rag.evaluate.generation.evaluate_answer") as answer_mock,
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark=benchmark,
                retriever=retriever,
                rerank=reranker,
            )

        answer_mock.assert_not_called()
        self.assertEqual(report["answer_accuracy"], 0.0)
        self.assertEqual(report["avg_score"], 0.0)
        self.assertEqual(report["citation_accuracy"], 0.0)
        self.assertEqual(report["retrieval_coverage"], 1.0)
        self.assertEqual(report["retrieval_recall"], 1.0)
        self.assertEqual(report["rerank_recall"], 1.0)
        self.assertEqual(report["rerank_coverage"], 1.0)
        self.assertEqual(report["generation_workers"], 1)
        self.assertEqual(report["evidence_generation_failed"], 0)
        self.assertEqual(report["answer_evaluation_failed"], 0)

    def test_evaluate_generation_returns_sample_details(self):
        docs = [
            {"node_id": "node-1", "content": "alpha", "rerank_score": 0.95},
            {"node_id": "node-2", "content": "beta", "rerank_score": 0.90},
        ]
        retriever = FakeRetriever(docs)
        reranker = FakeReranker(docs)
        benchmark = [
            {
                "_id": "qa-1",
                "question": "What is the summary?",
                "answer": "Alpha and beta.",
                "node_ids": ["node-1", "node-2"],
            }
        ]
        generated = SimpleNamespace(
            evidence_summary="Alpha and beta.",
            citations=["node-1", "node-2"],
        )

        with (
            patch("src.rag.evaluate.generation.evaluate_evidence", return_value=generated),
            patch("src.rag.evaluate.generation.evaluate_answer", return_value=0.92),
        ):
            report = evaluate_generation(
                answer_llm="answer-llm",
                judge_llm="judge-llm",
                benchmark=benchmark,
                retriever=retriever,
                rerank=reranker,
                include_details=True,
            )

        self.assertIn("sample_details", report)
        self.assertEqual(len(report["sample_details"]), 1)
        detail = report["sample_details"][0]
        self.assertEqual(detail["sample_index"], 1)
        self.assertEqual(detail["qa_id"], "qa-1")
        self.assertEqual(detail["question"], "What is the summary?")
        self.assertEqual(detail["ground_truth_node_ids"], ["node-1", "node-2"])
        self.assertEqual(detail["retrieval_node_ids"], ["node-1", "node-2"])
        self.assertEqual(detail["retrieval_hit_count"], 2)
        self.assertEqual(detail["retrieval_recall"], 1.0)
        self.assertEqual(detail["retrieval_mrr"], 0.75)
        self.assertEqual(detail["retrieval_coverage"], 1)
        self.assertEqual(detail["rerank_node_ids"], ["node-1", "node-2"])
        self.assertEqual(detail["rerank_hit_count"], 2)
        self.assertEqual(detail["rerank_recall"], 1.0)
        self.assertEqual(detail["rerank_mrr"], 0.75)
        self.assertEqual(detail["rerank_coverage"], 1)
        self.assertEqual(detail["generated_answer"], "Alpha and beta.")
        self.assertEqual(detail["answer_score"], 0.92)
        self.assertTrue(detail["answer_passed"])
        self.assertEqual(detail["answer_citations"], ["node-1", "node-2"])
        self.assertEqual(detail["citation_hit_count"], 2)
        self.assertEqual(detail["citation_coverage"], 1)
        self.assertIsNone(detail["skipped_reason"])


if __name__ == "__main__":
    unittest.main()
