import os
import sys
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace

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


if "core.settings" not in sys.modules:
    settings_module = types.ModuleType("core.settings")
    settings_module.settings = SimpleNamespace(
        graph_enabled=True,
        retriever_top_k=5,
        reranker_top_k=3,
        metadata_version=1,
        graph_query_top_k=6,
        graph_query_max_candidates=60,
    )
    sys.modules["core.settings"] = settings_module

if "src.graph.planner" not in sys.modules:
    graph_planner_module = types.ModuleType("src.graph.planner")

    def _looks_like_financial_graph_query(text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in ["compare", "trend", "revenue", "profit"])

    graph_planner_module.looks_like_financial_graph_query = _looks_like_financial_graph_query
    sys.modules["src.graph.planner"] = graph_planner_module

if "langchain_core.messages" not in sys.modules:
    langchain_core_module = types.ModuleType("langchain_core")
    langchain_messages_module = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    langchain_messages_module.HumanMessage = HumanMessage
    sys.modules["langchain_core"] = langchain_core_module
    sys.modules["langchain_core.messages"] = langchain_messages_module

if "src.config.llm_config" not in sys.modules:
    llm_config_module = types.ModuleType("src.config.llm_config")

    class LLMService:
        @staticmethod
        def invoke(*args, **kwargs):
            raise AssertionError("LLMService.invoke should be patched in tests")

    llm_config_module.LLMService = LLMService
    sys.modules["src.config.llm_config"] = llm_config_module

if "src.models.llm" not in sys.modules:
    llm_module = types.ModuleType("src.models.llm")
    llm_module.chatgpt_llm = object()
    sys.modules["src.models.llm"] = llm_module

from src.agent.action_planner import choose_next_action
from src.agent.policy import get_allowed_actions
from src.nodes.agent_node import agent_node
from src.types.agent_state import State
from src.types.event_type import MemoryEvent, ReasoningEvent
from src.types.policy_type import AgentPlannerDecision, AgentPlannerStructuredDecision


def _build_initial_state(query: str) -> State:
    return State(
        query=query,
        resolved_query=query,
        working_query=query,
        action_history=[
            ReasoningEvent(name="resolved_query", status="success", attempt=1),
            MemoryEvent(name="memory_recall", status="success", attempt=1),
        ],
    )


class AgentPlannerTests(unittest.TestCase):
    def test_initial_graph_query_exposes_graph_rag_to_planner(self):
        state = _build_initial_state("Compare revenue between 2024 and 2025")

        allowed_actions = get_allowed_actions(state)

        self.assertEqual(allowed_actions[0], "graph_rag")
        self.assertIn("rag", allowed_actions)

    def test_planner_falls_back_when_llm_returns_invalid_action(self):
        state = _build_initial_state("Compare revenue between 2024 and 2025")

        with patch(
            "src.agent.action_planner.LLMService.invoke",
            return_value=AgentPlannerDecision(
                next_action="web_search",
                reason="invalid for this candidate set",
                confidence=0.2,
            ),
        ):
            decision = choose_next_action(state, ["graph_rag", "rag"], planning_stage="initial")

        self.assertEqual(decision.next_action, "graph_rag")
        self.assertEqual(decision.reason, "planner_invalid_action_fallback")
        self.assertIn("planner:invalid_action=web_search", decision.diagnostics)

    def test_planner_uses_structured_schema_without_runtime_metadata_fields(self):
        state = _build_initial_state("Compare revenue between 2024 and 2025")

        def fake_invoke(*, llm, messages, schema, fallback_llm=None):
            self.assertIs(schema, AgentPlannerStructuredDecision)
            return AgentPlannerDecision(
                next_action="graph_rag",
                reason="financial fact comparison fits graph retrieval",
                confidence=0.91,
            )

        with patch("src.agent.action_planner.LLMService.invoke", side_effect=fake_invoke):
            decision = choose_next_action(state, ["graph_rag", "rag"], planning_stage="initial")

        self.assertTrue(decision.success)
        self.assertEqual(decision.next_action, "graph_rag")

    def test_agent_node_uses_planner_result_for_initial_route(self):
        state = _build_initial_state("Compare revenue between 2024 and 2025")

        with patch(
            "src.nodes.agent_node.choose_next_action",
            return_value=AgentPlannerDecision(
                success=True,
                next_action="graph_rag",
                reason="financial fact comparison fits graph retrieval",
                confidence=0.91,
                diagnostics=["planner:selected=graph_rag"],
            ),
        ):
            result = agent_node(state)

        self.assertEqual(result["action"], "graph_rag")
        self.assertEqual(result["reason"], "financial fact comparison fits graph retrieval")
        self.assertIn("agent:planner=graph_rag", result["diagnostics"])


if __name__ == "__main__":
    unittest.main()
