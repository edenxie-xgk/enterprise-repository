from langchain_core.messages import HumanMessage

from src.congfig.build_agent_context import build_agent_context
from src.congfig.llm_config import LLMService
from src.models.llm import deepseek_llm
from src.prompts.agent.agent import AGENT_PROMPT
from src.router.index import route_list
from src.types.agent_state import State
from src.types.base_type import BaseLLMDecideResult


def agent_node(state: State):
    # ===== 1️⃣ 构建上下文 =====
    last_event = state.tool_history[-1] if state.tool_history else []

    ctx = build_agent_context(last_event) if last_event else {
        "tool_name": None,
        "query":"",
        "answer": "",
        "fail_reason":"",
        "evidence": [],
        "quality_hint": {}
    }

    print(state)
    # ===== 4️⃣ 构建 Prompt =====
    prompt = AGENT_PROMPT.format(
        query=state.query,
        working_query=state.working_query or  state.query,
        tool_history=state.tool_history,
        last_action=state.last_action,
        last_tool_query=ctx['query'],
        last_tool_name=ctx['tool_name'],
        last_tool_answer=ctx["answer"],
        last_tool_evidence=ctx["evidence"],
        last_tool_quality_hint=ctx["quality_hint"],
        last_tool_fail_reason=ctx["fail_reason"],
        rewrite_attempt=state.rewrite_attempt,
        query_used=state.query_used
    )

    # ===== 5️⃣ 调用 LLM =====
    llm_result:BaseLLMDecideResult = LLMService.invoke(
        llm=deepseek_llm,
        messages=[HumanMessage(content=prompt)],
        # 建议你用 schema（非常重要）
        schema=BaseLLMDecideResult
    )



    # ===== 6️⃣ 防御（企业级必须有）=====
    action = llm_result.next_action

    if action not in route_list:
        action = "finish"


    # 防止死循环（同一工具限制最大调用次数，默认是三次）
    if any(True for item in state.tool_history if item.tool_name == action and item.output.attempt >= item.output.max_attempt) or (state.rewrite_attempt >= 2 and action=='rewrite_query'):
        action = "finish"

    return {
        "action": action,
        "last_action": action,
        "query_used": True,
        "working_query":state.working_query or  state.query,
    }