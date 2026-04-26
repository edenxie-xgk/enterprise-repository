from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agent.action_registry import dedupe_action_names, render_action_catalog
from src.config.llm_config import LLMService
from src.models.llm import chatgpt_llm
from src.prompts.agent.agent import AGENT_PROMPT
from src.prompts.agent.initial_action import INITIAL_ACTION_PROMPT
from src.types.agent_state import State
from src.types.policy_type import AgentPlannerDecision, AgentPlannerStructuredDecision


def _format_chat_history(chat_history: list[str]) -> str:
    if not chat_history:
        return "None"
    return "\n".join(f"- {item}" for item in chat_history[-8:])


def _format_query_evolution(state: State) -> str:
    lines = [
        f"- raw_query: {state.query or 'None'}",
        f"- resolved_query: {state.resolved_query or 'None'}",
        f"- working_query: {state.working_query or 'None'}",
    ]
    if state.rewrite_query:
        lines.append(f"- rewrite_query: {state.rewrite_query}")
    if state.expand_query:
        lines.append(f"- expand_query: {', '.join(state.expand_query[:5])}")
    if state.decompose_query:
        lines.append(f"- decompose_query: {', '.join(state.decompose_query[:5])}")
    return "\n".join(lines)


def _format_recent_context(state: State) -> str:
    lines: list[str] = []
    for event in state.action_history[-6:]:
        parts = [f"{event.name} ({event.kind}, status={event.status}, attempt={event.attempt or 0})"]
        if event.output:
            fail_reason = getattr(event.output, "fail_reason", None)
            message = getattr(event.output, "message", None) or getattr(event.output, "reason", None)
            if fail_reason:
                parts.append(f"fail_reason={fail_reason}")
            if message:
                parts.append(f"note={message}")
        lines.append("- " + "; ".join(parts))

    if state.last_rag_result:
        lines.append(
            "- last_rag_result: "
            f"is_sufficient={bool(state.last_rag_result.is_sufficient)}; "
            f"documents={len(state.last_rag_result.documents or [])}; "
            f"fail_reason={getattr(state.last_rag_result, 'fail_reason', None) or 'None'}"
        )
    if state.last_graph_result:
        lines.append(
            "- last_graph_result: "
            f"is_sufficient={bool(state.last_graph_result.is_sufficient)}; "
            f"documents={len(state.last_graph_result.documents or [])}; "
            f"fail_reason={getattr(state.last_graph_result, 'fail_reason', None) or 'None'}"
        )
    return "\n".join(lines) if lines else "None"


def _build_planner_prompt(state: State, allowed_actions: list[str], *, planning_stage: str) -> str:
    query = state.working_query or state.resolved_query or state.query or ""
    action_catalog = render_action_catalog(allowed_actions)

    if planning_stage == "initial":
        return INITIAL_ACTION_PROMPT.format(
            raw_query=state.query or "",
            query=query,
            chat_history=_format_chat_history(state.chat_history),
            allowed_actions=", ".join(allowed_actions),
            action_catalog=action_catalog,
        ).strip()

    return AGENT_PROMPT.format(
        raw_query=state.query or "",
        query=query,
        query_evolution=_format_query_evolution(state),
        context=_format_recent_context(state),
        allowed_actions=", ".join(allowed_actions),
        action_catalog=action_catalog,
    ).strip()


def choose_next_action(
    state: State,
    allowed_actions: list[str],
    *,
    planning_stage: str = "followup",
) -> AgentPlannerDecision:
    normalized_actions = dedupe_action_names(allowed_actions)
    fallback_action = normalized_actions[0] if normalized_actions else "finish"

    if not normalized_actions:
        return AgentPlannerDecision(
            success=False,
            next_action="finish",
            reason="planner_no_allowed_actions",
            confidence=0.0,
            diagnostics=["planner:no_allowed_actions"],
        )

    if len(normalized_actions) == 1:
        return AgentPlannerDecision(
            success=True,
            next_action=fallback_action,
            reason="single_allowed_action",
            confidence=1.0,
            diagnostics=[f"planner:single_allowed_action={fallback_action}"],
        )

    prompt = _build_planner_prompt(state, normalized_actions, planning_stage=planning_stage)
    try:
        payload = LLMService.invoke(
            llm=chatgpt_llm,
            messages=[HumanMessage(content=prompt)],
            schema=AgentPlannerStructuredDecision,
        )
        decision = AgentPlannerDecision(
            next_action=(getattr(payload, "next_action", None) or "").strip() or fallback_action,
            reason=getattr(payload, "reason", None),
            confidence=getattr(payload, "confidence", None),
            clarification_question=getattr(payload, "clarification_question", None),
        )
    except Exception as exc:
        return AgentPlannerDecision(
            success=False,
            next_action=fallback_action,
            reason="planner_llm_failed_fallback",
            confidence=0.0,
            error_detail=str(exc),
            diagnostics=[
                f"planner:stage={planning_stage}",
                f"planner:llm_failed_fallback={fallback_action}",
            ],
        )

    chosen_action = (decision.next_action or "").strip()
    if chosen_action not in normalized_actions:
        return AgentPlannerDecision(
            success=False,
            next_action=fallback_action,
            reason="planner_invalid_action_fallback",
            confidence=0.0,
            diagnostics=[
                f"planner:stage={planning_stage}",
                f"planner:invalid_action={chosen_action or 'empty'}",
                f"planner:fallback={fallback_action}",
            ],
        )

    decision.success = True
    decision.next_action = chosen_action
    decision.reason = (decision.reason or "planner_selected_action").strip()
    decision.diagnostics = list(decision.diagnostics or []) + [
        f"planner:stage={planning_stage}",
        f"planner:selected={chosen_action}",
    ]

    if decision.next_action == "clarify_question" and not (decision.clarification_question or "").strip():
        decision.clarification_question = "Please clarify the exact subject, scope, or period you want to ask about."

    return decision
