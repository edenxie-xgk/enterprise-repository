import os
import unittest
from unittest.mock import patch

_TEST_ENV = {
    "DELETE_FILE": "false",
    "DATABASE_NAME": "test_db",
    "DATABASE_STRING": "postgresql://user:pass@localhost:5432/test_db",
    "DATABASE_ASYNC_STRING": "postgresql+asyncpg://user:pass@localhost:5432/test_db",
    "VECTOR_TABLE_NAME": "vectors",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "EMBEDDING_DIM": "1536",
    "MONGODB_URL": "mongodb://localhost:27017",
    "MONGODB_DB_NAME": "test_db",
    "DOC_COLLECTION_NAME": "docs",
    "QA_COLLECTION_NAME": "qa",
    "ELASTICSEARCH_URL": "http://localhost:9200",
    "OPENAI_API_KEY": "test-openai-key",
    "OPENAI_MODEL": "gpt-4o-mini",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "DEEPSEEK_URL": "https://api.deepseek.com",
    "DEEPSEEK_MODEL": "deepseek-chat",
    "DEEPSEEK_API_KEY": "test-deepseek-key",
    "GRAPH_ENABLED": "true",
    "GRAPH_ENTITY_COLLECTION_NAME": "graph_entities",
    "GRAPH_FACT_COLLECTION_NAME": "graph_facts",
    "GRAPH_MAX_FACTS_PER_CHUNK": "12",
    "GRAPH_QUERY_TOP_K": "6",
    "GRAPH_QUERY_MAX_CANDIDATES": "60",
    "MAX_RETRIES": "1",
    "MAX_TIMEOUT": "30",
}

for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)

from core.custom_types import DocumentMetadata
from src.agent.policy import _looks_like_graph_query
from src.graph.extractor import FinancialFactExtractor
from src.graph.planner import interpret_financial_graph_query
from src.graph.ranking import score_graph_fact, select_diverse_facts
from src.types.graph_type import GraphQueryContext


class FinancialGraphExtractorTests(unittest.TestCase):
    def test_extractor_extracts_metric_and_event_facts(self):
        extractor = FinancialFactExtractor()
        metadata = DocumentMetadata(
            file_name="Interim Financial Statements for the period ended 30 September 2025.pdf",
            file_path="/public/uploads/TQ/report.pdf",
            file_type="pdf",
            department_id=2,
            department_name="TQ",
            page=12,
            chunk_index=3,
            section_title="Capital Commitments and Contingencies",
        )
        chunk_text = """
        SOFTLOGIC LIFE INSURANCE PLC
        13 CAPITAL COMMITMENTS AND CONTINGENCIES
        Profit before tax 3,279,719
        Dividend paid 726,152
        The Company acquired 100% of the issued share capital of Softlogic Life Insurance Lanka Limited.
        The purchase consideration of Rs. 1,426 Million was paid in cash.
        """

        result = extractor.extract_chunk(
            node_id="node-1",
            text=chunk_text,
            metadata=metadata,
        )

        self.assertGreaterEqual(len(result.facts), 3)
        metric_names = {fact.metric_name for fact in result.facts if fact.fact_kind == "metric"}
        topics = {fact.topic for fact in result.facts}
        self.assertIn("profit_before_tax", metric_names)
        self.assertIn("dividend_paid", metric_names)
        self.assertIn("acquisition", topics)


class FinancialGraphPlanningTests(unittest.TestCase):
    def test_interpret_query_marks_metric_comparison(self):
        interpretation = interpret_financial_graph_query(
            "Compare profit before tax and revenue trends between 2024 and 2025"
        )

        self.assertEqual(interpretation.query_kind, "period_comparison")
        self.assertIn("profit_before_tax", interpretation.metric_names)
        self.assertIn("revenue", interpretation.metric_names)
        self.assertEqual(interpretation.years, ["2024", "2025"])
        self.assertTrue(interpretation.comparison_mode)

    def test_policy_detects_graph_query(self):
        with patch("src.agent.policy.settings.graph_enabled", True):
            self.assertTrue(_looks_like_graph_query("Compare net profit trends between 2024 and 2025"))
            self.assertTrue(_looks_like_graph_query("What related party transactions are disclosed by the company"))
            self.assertFalse(_looks_like_graph_query("Which files did I upload recently"))


class FinancialGraphRankingTests(unittest.TestCase):
    def test_score_prefers_requested_metric(self):
        context = GraphQueryContext(
            query="Compare revenue between 2024 and 2025",
            query_kind="period_comparison",
            metric_names=["revenue"],
            years=["2024", "2025"],
            search_terms=["compare", "revenue", "2024", "2025"],
            comparison_mode=True,
            top_k=2,
            max_candidate_facts=6,
        )
        revenue_fact = {
            "fact_kind": "metric",
            "normalized_metric_name": "revenue",
            "topic": "financial_performance",
            "period_year": "2025",
            "normalized_company_name": "softlogic_life_insurance_plc",
            "search_terms": ["revenue", "income"],
            "confidence": 0.9,
            "numeric_value": 120.0,
            "evidence_node_ids": ["node-2025"],
        }
        unrelated_fact = {
            "fact_kind": "event",
            "normalized_metric_name": "",
            "topic": "acquisition",
            "period_year": "2025",
            "normalized_company_name": "softlogic_life_insurance_plc",
            "search_terms": ["acquisition"],
            "confidence": 0.4,
            "evidence_node_ids": ["node-acq"],
        }

        self.assertGreater(score_graph_fact(revenue_fact, context), score_graph_fact(unrelated_fact, context))

    def test_diversification_prefers_multiple_years_in_comparison_mode(self):
        context = GraphQueryContext(
            query="Compare profit before tax between 2024 and 2025",
            query_kind="period_comparison",
            metric_names=["profit_before_tax"],
            years=["2024", "2025"],
            search_terms=["compare", "profit_before_tax", "2024", "2025"],
            comparison_mode=True,
            top_k=2,
            max_candidate_facts=6,
        )
        rows = [
            {
                "fact_kind": "metric",
                "normalized_metric_name": "profit_before_tax",
                "topic": "financial_performance",
                "period_year": "2025",
                "search_terms": ["profit_before_tax", "2025"],
                "confidence": 0.95,
                "numeric_value": 3279.0,
                "evidence_node_ids": ["node-2025-a"],
            },
            {
                "fact_kind": "metric",
                "normalized_metric_name": "profit_before_tax",
                "topic": "financial_performance",
                "period_year": "2025",
                "search_terms": ["profit_before_tax", "2025"],
                "confidence": 0.94,
                "numeric_value": 3278.0,
                "evidence_node_ids": ["node-2025-b"],
            },
            {
                "fact_kind": "metric",
                "normalized_metric_name": "profit_before_tax",
                "topic": "financial_performance",
                "period_year": "2024",
                "search_terms": ["profit_before_tax", "2024"],
                "confidence": 0.88,
                "numeric_value": 3010.0,
                "evidence_node_ids": ["node-2024-a"],
            },
        ]

        selected = select_diverse_facts(rows, context, top_k=2)
        selected_years = {row.get("period_year") for row in selected}
        self.assertEqual(selected_years, {"2024", "2025"})


if __name__ == "__main__":
    unittest.main()
