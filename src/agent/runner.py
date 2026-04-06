from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import uuid4

from src.agent.graph import graph
from src.types.agent_state import State


def run_agent(
    query: str,
    *,
    user_id: str = "",
    session_id: str = "",
    user_profile: dict[str, Any] | None = None,
    chat_history: list[str] | None = None,
    max_steps: int = 20,
) -> State:
    """Run the agent graph and return the final state."""
    initial_state = State(
        query=query,
        run_id=str(uuid4()),
        user_id=user_id,
        session_id=session_id,
        user_profile=user_profile,
        chat_history=chat_history or [],
        max_steps=max_steps,
    )
    result = graph.invoke(initial_state)
    if isinstance(result, State):
        return result
    return State(**result)


def summarize_trace(state: State) -> list[dict[str, Any]]:
    """Convert trace records into a JSON-friendly list."""
    rows = []
    for item in state.trace:
        rows.append(
            {
                "step": item.step,
                "event": item.event_name,
                "kind": item.event_kind,
                "status": item.status,
                "attempt": item.attempt,
                "duration_ms": item.duration_ms,
                "fail_reason": item.fail_reason,
                "message": item.message,
                "diagnostics": item.diagnostics,
            }
        )
    return rows


def build_run_report(state: State) -> dict[str, Any]:
    last_rag_result = state.last_rag_result
    sub_query_results = []
    for item in state.sub_query_results:
        if hasattr(item, "model_dump"):
            sub_query_results.append(item.model_dump())
        elif isinstance(item, dict):
            sub_query_results.append(item)

    return {
        "run_id": state.run_id,
        "query": state.query,
        "status": state.status,
        "action": state.action,
        "answer": state.answer,
        "citations": state.citations,
        "reason": state.reason,
        "fail_reason": state.fail_reason,
        "current_step": state.current_step,
        "max_steps": state.max_steps,
        "working_memory": state.working_memory,
        "short_term_memory": state.short_term_memory,
        "diagnostics": state.diagnostics,
        "rewrite_query": state.rewrite_query,
        "expand_query": state.expand_query,
        "decompose_query": state.decompose_query,
        "sub_query_results": sub_query_results,
        "last_rag_result": {
            "answer": getattr(last_rag_result, "answer", ""),
            "evidence_summary": getattr(last_rag_result, "evidence_summary", ""),
            "is_sufficient": getattr(last_rag_result, "is_sufficient", False),
            "fail_reason": getattr(last_rag_result, "fail_reason", None),
            "confidence": getattr(last_rag_result, "confidence", None),
            "citations": getattr(last_rag_result, "citations", []),
            "retrieval_queries": getattr(last_rag_result, "retrieval_queries", []),
            "retrieval_candidate_node_ids": getattr(last_rag_result, "retrieval_candidate_node_ids", []),
            "rerank_node_ids": getattr(last_rag_result, "rerank_node_ids", []),
            "diagnostics": getattr(last_rag_result, "diagnostics", []),
            "document_count": len(getattr(last_rag_result, "documents", []) or []),
        }
        if last_rag_result
        else None,
        "trace": summarize_trace(state),
    }


def print_run_report(state: State) -> None:
    print(json.dumps(build_run_report(state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agentic RAG graph for local validation.")
    parser.add_argument(
        "query",
        nargs="?",
        default="根据多个会议记录，不同部门对新能源行业景气度的看法有哪些共同点和差异？",
        help="User query",
    )
    parser.add_argument("--user-id", default="", help="User ID")
    parser.add_argument("--session-id", default="", help="Session ID")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum agent steps")
    args = parser.parse_args()

    final_state = run_agent(
        args.query,
        user_id=args.user_id,
        session_id=args.session_id,
        max_steps=args.max_steps,
    )
    print_run_report(final_state)
