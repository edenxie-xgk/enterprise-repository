import sys
import types
import unittest
from unittest.mock import patch


if "core.settings" not in sys.modules:
    settings_module = types.ModuleType("core.settings")
    settings_module.settings = types.SimpleNamespace()
    sys.modules["core.settings"] = settings_module

if "langchain_core.language_models" not in sys.modules:
    language_models_module = types.ModuleType("langchain_core.language_models")

    class BaseChatModel:
        pass

    language_models_module.BaseChatModel = BaseChatModel
    sys.modules["langchain_core.language_models"] = language_models_module

if "langchain_core.messages" not in sys.modules:
    messages_module = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    messages_module.HumanMessage = HumanMessage
    sys.modules["langchain_core.messages"] = messages_module

if "src.models.llm" not in sys.modules:
    llm_module = types.ModuleType("src.models.llm")
    llm_module.deepseek_llm = object()
    sys.modules["src.models.llm"] = llm_module

if "src.config.llm_config" not in sys.modules:
    llm_config_module = types.ModuleType("src.config.llm_config")

    class LLMService:
        @staticmethod
        def invoke(*args, **kwargs):
            raise AssertionError("LLMService.invoke should be patched in tests")

    llm_config_module.LLMService = LLMService
    sys.modules["src.config.llm_config"] = llm_config_module

if "src.prompts.rag.qa_generation" not in sys.modules:
    prompt_module = types.ModuleType("src.prompts.rag.qa_generation")
    prompt_module.QA_GENERATION_PROMPT = "nodes={nodes}"
    sys.modules["src.prompts.rag.qa_generation"] = prompt_module

if "utils.utils" not in sys.modules:
    utils_module = types.ModuleType("utils.utils")
    utils_module.get_current_time = lambda: "2026-04-25 00:00:00"
    sys.modules["utils.utils"] = utils_module

from src.rag.evaluate.qa import QAResult, QaData, generate_qa


class QAGenerationTests(unittest.TestCase):
    def test_single_qa_result_does_not_raise_and_defaults_to_state_zero(self):
        response = QAResult(
            qa_list=[
                QaData(
                    question="问题1",
                    answer="答案1",
                    language="zh",
                    difficulty="easy",
                    intent="fact",
                    node_ids=["node-1"],
                )
            ]
        )

        with patch("src.rag.evaluate.qa.LLMService.invoke", return_value=response):
            rows = generate_qa(llm=object(), nodes=[{"node_id": "node-1", "content": "内容"}], metadata={"dept": 1})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], 0)
        self.assertEqual(rows[0]["metadata"], {"dept": 1})

    def test_multiple_qa_results_assign_exactly_one_validation_sample(self):
        response = QAResult(
            qa_list=[
                QaData(
                    question="问题1",
                    answer="答案1",
                    language="zh",
                    difficulty="easy",
                    intent="fact",
                    node_ids=["node-1"],
                ),
                QaData(
                    question="问题2",
                    answer="答案2",
                    language="zh",
                    difficulty="medium",
                    intent="compare",
                    node_ids=["node-1", "node-2"],
                ),
                QaData(
                    question="问题3",
                    answer="答案3",
                    language="zh",
                    difficulty="hard",
                    intent="summary",
                    node_ids=["node-2"],
                ),
            ]
        )

        nodes = [
            {"node_id": "node-1", "content": "内容1"},
            {"node_id": "node-2", "content": "内容2"},
        ]

        with patch("src.rag.evaluate.qa.LLMService.invoke", return_value=response):
            with patch("src.rag.evaluate.qa.random.randrange", return_value=1):
                rows = generate_qa(llm=object(), nodes=nodes, metadata=None)

        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(1 for row in rows if row["state"] == 2), 1)
        self.assertEqual(rows[1]["state"], 2)


if __name__ == "__main__":
    unittest.main()
