from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence


ActionCategory = Literal["tool", "reasoning", "terminal", "response"]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    category: ActionCategory
    summary: str
    when_to_use: str
    when_not_to_use: str = ""


ACTION_REGISTRY: dict[str, ActionSpec] = {
    "rag": ActionSpec(
        name="rag",
        category="tool",
        summary="Retrieve narrative evidence from the enterprise knowledge base.",
        when_to_use="Use for internal documents, reports, policies, uploaded files, and general document-grounded QA.",
        when_not_to_use="Do not use for structured field lookups, public real-time information, or finance fact-graph reasoning.",
    ),
    "graph_rag": ActionSpec(
        name="graph_rag",
        category="tool",
        summary="Query the financial fact graph for metrics, entity relations, and cross-report facts.",
        when_to_use="Use for finance metric comparison, trend analysis, entity-event relations, related-party transactions, or fact linking across reports.",
        when_not_to_use="Do not use for generic narrative document QA or non-financial internal policy questions.",
    ),
    "web_search": ActionSpec(
        name="web_search",
        category="tool",
        summary="Retrieve public and time-sensitive information from the web.",
        when_to_use="Use for latest news, recent market updates, current policy changes, and external public information.",
        when_not_to_use="Do not use for internal documents, uploaded files, or private enterprise data.",
    ),
    "db_search": ActionSpec(
        name="db_search",
        category="tool",
        summary="Query structured internal records and fields.",
        when_to_use="Use for counts, lists, permissions, ownership mappings, upload records, and structured business fields.",
        when_not_to_use="Do not use for open-ended narrative QA that needs document evidence.",
    ),
    "rewrite_query": ActionSpec(
        name="rewrite_query",
        category="reasoning",
        summary="Rewrite the query into a cleaner retrieval-oriented form.",
        when_to_use="Use when the wording is vague, colloquial, underspecified, or overly dependent on prior chat context.",
        when_not_to_use="Do not use when the query is already precise and ready for retrieval.",
    ),
    "expand_query": ActionSpec(
        name="expand_query",
        category="reasoning",
        summary="Broaden the query to improve recall.",
        when_to_use="Use when recall is too narrow and the system needs more related search terms.",
        when_not_to_use="Do not use when precision matters more than recall or the query is already broad enough.",
    ),
    "decompose_query": ActionSpec(
        name="decompose_query",
        category="reasoning",
        summary="Split a complex request into smaller sub-questions.",
        when_to_use="Use for multi-part analysis, comparison, stepwise planning, or requests with multiple explicit objectives.",
        when_not_to_use="Do not use for short and focused single-goal questions.",
    ),
    "direct_answer": ActionSpec(
        name="direct_answer",
        category="response",
        summary="Answer directly without calling retrieval tools.",
        when_to_use="Use when the question can be answered safely from general knowledge and chat context without enterprise or real-time evidence.",
        when_not_to_use="Do not use for internal knowledge retrieval, structured data lookup, or time-sensitive external questions.",
    ),
    "clarify_question": ActionSpec(
        name="clarify_question",
        category="response",
        summary="Ask the user to clarify missing scope or subject.",
        when_to_use="Use when the request is too incomplete or ambiguous to continue safely.",
        when_not_to_use="Do not use if the system can reasonably proceed with an informed assumption.",
    ),
    "finalize": ActionSpec(
        name="finalize",
        category="terminal",
        summary="Draft the final grounded answer from retrieved evidence.",
        when_to_use="Use when enough evidence has been gathered to synthesize a response.",
        when_not_to_use="Do not use when the evidence gap is still significant.",
    ),
    "finish": ActionSpec(
        name="finish",
        category="terminal",
        summary="Stop the workflow and return the best available answer or fallback response.",
        when_to_use="Use when no higher-value next action remains or the workflow is exhausted.",
        when_not_to_use="Do not use when another allowed action is likely to add meaningful evidence.",
    ),
    "abort": ActionSpec(
        name="abort",
        category="terminal",
        summary="Terminate the run without further action.",
        when_to_use="Use only for hard-stop or invalid workflow states.",
        when_not_to_use="Do not use as a normal answer path.",
    ),
}


REASONING_ACTION_NAMES: tuple[str, ...] = (
    "rewrite_query",
    "expand_query",
    "decompose_query",
)

TOOL_ACTION_NAMES: tuple[str, ...] = (
    "rag",
    "graph_rag",
    "web_search",
    "db_search",
)

TERMINAL_ACTION_NAMES: tuple[str, ...] = (
    "finish",
    "abort",
)

INITIAL_ACTION_NAMES: tuple[str, ...] = (
    "rag",
    "graph_rag",
    "db_search",
    "web_search",
    "rewrite_query",
    "decompose_query",
    "clarify_question",
    "direct_answer",
)

ROUTE_ACTION_NAMES: tuple[str, ...] = (
    *REASONING_ACTION_NAMES,
    *TOOL_ACTION_NAMES,
    "finalize",
    "abort",
    "finish",
    "direct_answer",
)


def dedupe_action_names(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def get_action_spec(action_name: str) -> ActionSpec:
    return ACTION_REGISTRY[action_name]


def render_action_catalog(action_names: Sequence[str]) -> str:
    lines: list[str] = []
    for action_name in dedupe_action_names(action_names):
        spec = get_action_spec(action_name)
        lines.append(
            f"- {spec.name}: {spec.summary} Use when: {spec.when_to_use} Avoid when: {spec.when_not_to_use or 'n/a'}"
        )
    return "\n".join(lines) if lines else "None"
