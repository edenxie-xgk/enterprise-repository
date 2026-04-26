from src.agent.action_planner import choose_next_action
from src.agent.policy import get_allowed_actions, guard_input, should_force_finish
from src.types.agent_state import State


DISALLOWED_INPUT_REASONS = {
    "illegal_cyber_activity",
    "privacy_exfiltration",
    "illegal_deception_request",
}


def _has_finalize_material(state: State) -> bool:
    last_rag_result = state.last_rag_result
    if last_rag_result:
        if last_rag_result.documents:
            return True
        if last_rag_result.evidence_summary or last_rag_result.answer:
            return True
    return bool(state.sub_query_results)


def _can_finalize(state: State, *, allow_partial: bool = False) -> bool:
    last_rag_result = state.last_rag_result
    if not last_rag_result:
        return False

    has_summary = bool((last_rag_result.evidence_summary or last_rag_result.answer or "").strip())
    has_citations = bool(last_rag_result.citations)
    has_documents = bool(last_rag_result.documents)
    is_sufficient = bool(last_rag_result.is_sufficient)
    fail_reason = getattr(last_rag_result, "fail_reason", None)

    if is_sufficient and has_summary and has_citations and has_documents:
        return True

    if not allow_partial:
        return False

    return has_summary and (has_citations or has_documents) and fail_reason in {
        "insufficient_context",
        "ambiguous_query",
    }


def _build_finish_response(
    state: State,
    *,
    answer: str,
    reason: str = "",
    status: str = "success",
    fail_reason=None,
    diagnostics=None,
):
    return {
        "action": "finish",
        "answer": answer,
        "reason": reason,
        "status": status,
        "fail_reason": fail_reason,
        "diagnostics": diagnostics or state.diagnostics,
    }


def _build_clarify_response(state: State, question: str, reason: str, diagnostics: list[str]):
    return _build_finish_response(
        state,
        answer=question or "Please clarify the exact subject, scope, or time period you want to ask about.",
        reason=reason or "clarify_question",
        status="success",
        fail_reason="ambiguous_query",
        diagnostics=diagnostics,
    )


def _route_with_planner(
    state: State,
    allowed_actions: list[str],
    *,
    planning_stage: str,
    default_reason: str,
):
    decision = choose_next_action(state, allowed_actions, planning_stage=planning_stage)
    chosen_action = decision.next_action or (allowed_actions[0] if allowed_actions else "finish")
    diagnostics = state.diagnostics + list(decision.diagnostics or [])

    if chosen_action == "finish" and _can_finalize(state, allow_partial=True):
        chosen_action = "finalize"
        diagnostics.append("agent:planner_promoted_finish_to_finalize")

    if chosen_action == "clarify_question":
        return _build_clarify_response(
            state,
            decision.clarification_question or "",
            decision.reason or default_reason,
            diagnostics + ["agent:planner=clarify_question"],
        )

    return {
        "action": chosen_action,
        "reason": decision.reason or default_reason,
        "diagnostics": diagnostics + [f"agent:planner={chosen_action}"],
    }


def agent_node(state: State):
    last_tool = next((event for event in reversed(state.action_history) if event.kind == "tool"), None)
    last_event = state.action_history[-1] if state.action_history else None
    effective_query = state.working_query or state.resolved_query or state.query or ""

    input_guard = guard_input(state.query or effective_query)
    if not input_guard.is_valid:
        return _build_finish_response(
            state,
            answer=input_guard.response or "",
            reason=input_guard.reason or "invalid_input",
            status="failed",
            fail_reason="disallowed_query" if input_guard.reason in DISALLOWED_INPUT_REASONS else "invalid_input",
            diagnostics=state.diagnostics + [f"agent:input_blocked:{input_guard.reason}"],
        )

    if state.current_step >= state.max_steps:
        if _can_finalize(state, allow_partial=True):
            return {
                "action": "finalize",
                "reason": f"Reached max steps: {state.max_steps}",
                "diagnostics": state.diagnostics + ["agent:max_steps_finalize"],
            }
        return _build_finish_response(
            state,
            answer=get_last_answer(last_tool) or build_fallback_answer(state),
            reason=f"Reached max steps: {state.max_steps}",
            status="failed",
            fail_reason="max_steps_exceeded",
            diagnostics=state.diagnostics + ["agent:max_steps_exceeded"],
        )

    if (
        last_event
        and last_event.kind == "tool"
        and getattr(last_event.output, "is_sufficient", False)
        and _can_finalize(state, allow_partial=False)
    ):
        return {
            "action": "finalize",
            "reason": "retrieval_evidence_is_sufficient",
            "diagnostics": state.diagnostics + ["agent:finalize_after_sufficient_evidence"],
        }

    force_finish, finish_reason = should_force_finish(state)
    if force_finish:
        if _can_finalize(state, allow_partial=True):
            return {
                "action": "finalize",
                "reason": finish_reason,
                "diagnostics": state.diagnostics + [f"agent:force_finalize:{finish_reason}"],
            }
        return _build_finish_response(
            state,
            answer=get_last_answer(last_tool) or build_fallback_answer(state),
            reason=finish_reason or "force_finish",
            status="failed",
            fail_reason=getattr(state.last_rag_result, "fail_reason", None) or "insufficient_context",
            diagnostics=state.diagnostics + [f"agent:force_finish:{finish_reason}"],
        )

    if last_tool and last_tool.name == "db_search":
        if _has_finalize_material(state):
            return {
                "action": "finalize",
                "reason": "db_search_has_material",
                "diagnostics": state.diagnostics + ["agent:db_search_finalize"],
            }
        return _build_finish_response(
            state,
            answer=get_last_answer(last_tool) or build_fallback_answer(state),
            reason="db_search_without_material",
            status="failed",
            fail_reason=getattr(last_tool.output, "fail_reason", None) or "no_data",
            diagnostics=state.diagnostics + ["agent:db_search_finish"],
        )

    if last_tool and last_tool.name in {"rag", "web_search", "graph_rag"}:
        if _can_finalize(state, allow_partial=False):
            return {
                "action": "finalize",
                "reason": f"{last_tool.name}_has_sufficient_material",
                "diagnostics": state.diagnostics + [f"agent:{last_tool.name}_finalize"],
            }
        if _has_finalize_material(state):
            return {
                "action": "finalize",
                "reason": f"{last_tool.name}_has_partial_material",
                "diagnostics": state.diagnostics + [f"agent:{last_tool.name}_partial_finalize"],
            }

        return _route_with_planner(
            state,
            get_allowed_actions(state),
            planning_stage="followup",
            default_reason=f"{last_tool.name}_planner_route",
        )

    if last_event and last_event.kind == "reasoning":
        return _route_with_planner(
            state,
            get_allowed_actions(state),
            planning_stage="followup",
            default_reason="planner_after_reasoning",
        )

    return _route_with_planner(
        state,
        get_allowed_actions(state),
        planning_stage="initial" if not last_tool else "followup",
        default_reason="planner_default_route",
    )


def get_last_answer(last_tool) -> str:
    if last_tool and last_tool.output:
        return getattr(last_tool.output, "evidence_summary", None) or getattr(last_tool.output, "answer", "") or ""
    return ""


def build_fallback_answer(state: State) -> str:
    if state.sub_query_results:
        return "The workflow tried decomposed retrieval, but the current evidence is still not stable enough to answer the original request."
    if state.last_rag_result and state.last_rag_result.fail_reason == "no_data":
        return "The current knowledge sources did not return enough relevant evidence to support a reliable answer."
    return "The current evidence is insufficient to produce a stable answer."
