import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _stubbed_modules() -> dict[str, types.ModuleType]:
    langchain_core_module = types.ModuleType("langchain_core")
    langchain_messages_module = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    langchain_messages_module.HumanMessage = HumanMessage

    llm_config_module = types.ModuleType("src.config.llm_config")

    class LLMService:
        @staticmethod
        def invoke(*args, **kwargs):
            raise AssertionError("LLMService.invoke should not run in prompt policy tests")

        @staticmethod
        def stream_text(*args, **kwargs):
            raise AssertionError("LLMService.stream_text should not run in prompt policy tests")

    llm_config_module.LLMService = LLMService

    llm_module = types.ModuleType("src.models.llm")
    llm_module.chatgpt_llm = object()

    answer_stream_module = types.ModuleType("src.agent.answer_stream")
    answer_stream_module.get_answer_token_handler = lambda: None

    policy_module = types.ModuleType("src.agent.policy")
    policy_module._looks_like_external_query = lambda text: False
    policy_module._looks_like_structured_db_query = lambda text: False
    policy_module.is_complex_query = lambda text: False
    policy_module.needs_rewrite_first = lambda text: False

    profile_utils_module = types.ModuleType("src.agent.profile_utils")
    profile_utils_module.extract_preferred_topics = lambda profile: []

    helpers_module = types.ModuleType("src.nodes.helpers")
    helpers_module.build_state_patch = lambda *args, **kwargs: {}
    helpers_module.create_event = lambda *args, **kwargs: SimpleNamespace(attempt=1)
    helpers_module.finalize_event = lambda event, result, start_time: event

    event_type_module = types.ModuleType("src.types.event_type")

    class ReasoningEvent:
        pass

    event_type_module.ReasoningEvent = ReasoningEvent

    agent_state_module = types.ModuleType("src.types.agent_state")

    class State:
        pass

    agent_state_module.State = State

    final_answer_module = types.ModuleType("src.types.final_answer_type")

    class FinalAnswerResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    final_answer_module.FinalAnswerResult = FinalAnswerResult

    direct_answer_prompt_module = types.ModuleType("src.prompts.agent.direct_answer_prompt")
    direct_answer_prompt_module.DIRECT_ANSWER_PROMPT = (
        "Raw: {raw_query}\n"
        "Resolved: {query}\n"
        "Chat: {chat_history}\n"
        "Level: {output_level}\n"
        "Language: {preferred_language}\n"
        "Topics: {preferred_topics}"
    )
    direct_answer_prompt_module.DIRECT_ANSWER_STREAM_PROMPT = direct_answer_prompt_module.DIRECT_ANSWER_PROMPT

    finalize_prompt_module = types.ModuleType("src.prompts.agent.finalize_prompt")
    finalize_prompt_module.FINALIZE_PROMPT = (
        "Raw: {raw_query}\n"
        "Resolved: {query}\n"
        "Evidence: {evidence_summary}\n"
        "Sub-query: {sub_query_context}\n"
        "Citations: {available_citations}\n"
        "Level: {output_level}\n"
        "Language: {preferred_language}\n"
        "CitePref: {prefers_citations}\n"
        "Topics: {preferred_topics}"
    )
    finalize_prompt_module.FINALIZE_STREAM_PROMPT = (
        "Raw: {raw_query}\n"
        "Resolved: {query}\n"
        "Evidence: {evidence_summary}\n"
        "Sub-query: {sub_query_context}\n"
        "Level: {output_level}\n"
        "Language: {preferred_language}\n"
        "Topics: {preferred_topics}"
    )

    return {
        "langchain_core": langchain_core_module,
        "langchain_core.messages": langchain_messages_module,
        "src.config.llm_config": llm_config_module,
        "src.models.llm": llm_module,
        "src.agent.answer_stream": answer_stream_module,
        "src.agent.policy": policy_module,
        "src.agent.profile_utils": profile_utils_module,
        "src.nodes.helpers": helpers_module,
        "src.types.event_type": event_type_module,
        "src.types.agent_state": agent_state_module,
        "src.types.final_answer_type": final_answer_module,
        "src.prompts.agent.direct_answer_prompt": direct_answer_prompt_module,
        "src.prompts.agent.finalize_prompt": finalize_prompt_module,
    }


def _import_with_stubs(module_name: str):
    sys.modules.pop(module_name, None)
    with patch.dict(sys.modules, _stubbed_modules()):
        return importlib.import_module(module_name)


class MemoryAnswerPolicyTests(unittest.TestCase):
    def test_direct_answer_prompt_allows_answering_user_profile_questions_from_memory(self):
        module = _import_with_stubs("src.nodes.direct_answer_node")
        state = SimpleNamespace(
            query="你还记得我是做什么的吗？",
            resolved_query="你还记得我是做什么的吗？",
            working_query="",
            chat_history=[],
            output_level="standard",
            user_profile={"preferred_language": "zh-CN"},
            long_term_memory_context="[task context] 我是一名程序员",
        )

        prompt = module._build_direct_answer_prompt(state)

        self.assertIn("[Long-term Memory Context]", prompt)
        self.assertIn("answer directly from long-term memory", prompt)
        self.assertIn("user-provided profile facts, preferences, identity", prompt)
        self.assertIn("Do not use long-term memory as documentary evidence", prompt)

    def test_finalize_prompt_keeps_rag_evidence_authoritative_over_long_term_memory(self):
        module = _import_with_stubs("src.nodes.finalize_node")
        state = SimpleNamespace(
            query="我之前让你记住了什么？",
            user_profile={"preferred_language": "zh-CN", "prefers_citations": True},
            output_level="standard",
            sub_query_results=[],
            long_term_memory_context="[identity_fact] 我叫王磊",
        )

        prompt = module._build_finalize_prompt(
            state,
            effective_query="我之前让你记住了什么？",
            evidence_summary="No evidence summary available.",
            citations=[],
        )

        self.assertIn("[Long-term Memory Context]", prompt)
        self.assertIn("do not let long-term memory change, extend, or override the evidence-based conclusion", prompt)
        self.assertIn("Only use long-term memory content directly", prompt)
        self.assertIn("trust the current evidence", prompt)
        self.assertIn("trust the current chat context", prompt)


if __name__ == "__main__":
    unittest.main()
