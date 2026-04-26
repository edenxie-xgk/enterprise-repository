from __future__ import annotations

from src.types.graph_type import GraphQueryContext


QUERY_KIND_FACT_PREFERENCE: dict[str, set[str]] = {
    "metric_lookup": {"metric"},
    "period_comparison": {"metric", "management_view"},
    "event_lookup": {"event", "policy"},
    "risk_lookup": {"risk", "event"},
    "related_party_lookup": {"related_party", "event"},
    "general": {"metric", "event", "risk", "related_party", "policy", "management_view"},
}


def _normalized_set(values: list[str] | None) -> set[str]:
    return {str(item).strip().lower() for item in values or [] if str(item).strip()}


def _first_evidence_node_id(row: dict) -> str:
    evidence_node_ids = [str(item).strip() for item in row.get("evidence_node_ids") or [] if str(item).strip()]
    if evidence_node_ids:
        return evidence_node_ids[0]

    evidence_docs = row.get("evidence_docs") or []
    for evidence in evidence_docs:
        if not isinstance(evidence, dict):
            continue
        node_id = str(evidence.get("node_id") or "").strip()
        if node_id:
            return node_id
    return ""


def score_graph_fact(row: dict, query_context: GraphQueryContext) -> float:
    score = 0.0
    preferred_kinds = QUERY_KIND_FACT_PREFERENCE.get(query_context.query_kind, QUERY_KIND_FACT_PREFERENCE["general"])
    requested_metrics = _normalized_set(query_context.metric_names)
    requested_topics = _normalized_set(query_context.topics)
    requested_years = _normalized_set(query_context.years)
    requested_companies = _normalized_set(query_context.company_terms)
    requested_terms = _normalized_set(query_context.search_terms)

    fact_kind = str(row.get("fact_kind") or "")
    if fact_kind in preferred_kinds:
        score += 5.0

    metric_name = str(row.get("normalized_metric_name") or "").lower()
    if metric_name and metric_name in requested_metrics:
        score += 6.0

    topic = str(row.get("topic") or "").lower()
    if topic and topic in requested_topics:
        score += 4.0

    year = str(row.get("period_year") or "").lower()
    if year and year in requested_years:
        score += 2.5

    company = str(row.get("normalized_company_name") or "").lower()
    if company and company in requested_companies:
        score += 3.0

    search_terms = {str(item).strip().lower() for item in row.get("search_terms") or [] if str(item).strip()}
    shared_terms = search_terms.intersection(requested_terms)
    score += min(len(shared_terms), 6) * 1.1

    if query_context.comparison_mode and year:
        score += 1.0
    if row.get("numeric_value") is not None:
        score += 0.6
    if row.get("evidence_node_ids"):
        score += 0.8

    score += float(row.get("confidence") or 0.0)
    return score


def _is_near_duplicate(left: dict, right: dict) -> bool:
    left_metric = str(left.get("normalized_metric_name") or "").strip().lower()
    right_metric = str(right.get("normalized_metric_name") or "").strip().lower()
    left_year = str(left.get("period_year") or "").strip().lower()
    right_year = str(right.get("period_year") or "").strip().lower()
    left_topic = str(left.get("topic") or "").strip().lower()
    right_topic = str(right.get("topic") or "").strip().lower()
    left_evidence = _first_evidence_node_id(left)
    right_evidence = _first_evidence_node_id(right)

    same_metric_period = bool(left_metric and left_metric == right_metric and left_year and left_year == right_year)
    same_topic_evidence = bool(left_topic and left_topic == right_topic and left_evidence and left_evidence == right_evidence)
    return same_metric_period or same_topic_evidence


def _diversity_adjustment(row: dict, selected_rows: list[dict], query_context: GraphQueryContext) -> float:
    if not selected_rows:
        return 0.0

    adjustment = 0.0
    year = str(row.get("period_year") or "").strip().lower()
    metric = str(row.get("normalized_metric_name") or "").strip().lower()
    topic = str(row.get("topic") or "").strip().lower()
    evidence_node_id = _first_evidence_node_id(row)

    selected_years = {str(item.get("period_year") or "").strip().lower() for item in selected_rows}
    selected_metrics = {str(item.get("normalized_metric_name") or "").strip().lower() for item in selected_rows}
    selected_topics = {str(item.get("topic") or "").strip().lower() for item in selected_rows}
    selected_evidence = {_first_evidence_node_id(item) for item in selected_rows}

    if query_context.comparison_mode and year:
        adjustment += 2.4 if year not in selected_years else -0.9

    if metric:
        adjustment += 1.0 if metric not in selected_metrics else -0.35

    if topic:
        adjustment += 0.4 if topic not in selected_topics else -0.15

    if evidence_node_id:
        adjustment += 0.9 if evidence_node_id not in selected_evidence else -1.1

    if any(_is_near_duplicate(row, existing) for existing in selected_rows):
        adjustment -= 1.5

    return adjustment


def select_diverse_facts(rows: list[dict], query_context: GraphQueryContext, top_k: int) -> list[dict]:
    if top_k <= 0 or not rows:
        return []

    base_scores = {id(row): score_graph_fact(row, query_context) for row in rows}
    remaining = sorted(rows, key=lambda item: base_scores[id(item)], reverse=True)
    selected: list[dict] = []

    while remaining and len(selected) < top_k:
        best_row = max(
            remaining,
            key=lambda item: base_scores[id(item)] + _diversity_adjustment(item, selected, query_context),
        )
        selected.append(best_row)
        remaining.remove(best_row)

    return selected
