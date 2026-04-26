from __future__ import annotations

import re

from core.settings import settings
from src.graph.extractor import METRIC_ALIASES, SECTION_TOPICS, _normalize_slug, _tokenize_text
from src.types.graph_type import GraphQueryInterpretation


COMPARISON_MARKERS = [
    "compare",
    "comparison",
    "difference",
    "trend",
    "change",
    "versus",
    "同比",
    "环比",
    "对比",
    "比较",
    "趋势",
    "变化",
]
RELATED_PARTY_MARKERS = ["related party", "parent", "subsidiary", "关联方", "子公司", "母公司", "kmp"]
RISK_MARKERS = ["risk", "contingency", "capital commitment", "assessment", "appeal", "风险", "或有", "资本承诺"]
EVENT_MARKERS = ["acquisition", "acquired", "merger", "tax", "dividend", "收购", "并购", "分红", "税务"]
COMPANY_PATTERN = re.compile(
    r"([A-Z][A-Z&.,\-\s]{3,}(?:PLC|LIMITED|LTD|INC|CORP))|([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})"
)


def looks_like_financial_graph_query(text: str) -> bool:
    if not settings.graph_enabled:
        return False
    interpretation = interpret_financial_graph_query(text)
    return bool(
        interpretation.metric_names
        or interpretation.topics
        or interpretation.company_terms
        or interpretation.query_kind != "general"
    )


def interpret_financial_graph_query(text: str) -> GraphQueryInterpretation:
    lowered = (text or "").lower()
    metric_names = [
        metric_name
        for metric_name, aliases in METRIC_ALIASES.items()
        if any(alias in lowered for alias in aliases)
    ]
    topics = [
        topic
        for topic, aliases in SECTION_TOPICS.items()
        if any(alias in lowered for alias in aliases)
    ]
    years = sorted(set(re.findall(r"\b20\d{2}\b", text or "")))
    comparison_mode = any(marker in lowered for marker in COMPARISON_MARKERS)

    if comparison_mode and metric_names:
        query_kind = "period_comparison"
    elif metric_names:
        query_kind = "metric_lookup"
    elif any(marker in lowered for marker in RELATED_PARTY_MARKERS):
        query_kind = "related_party_lookup"
    elif any(marker in lowered for marker in RISK_MARKERS):
        query_kind = "risk_lookup"
    elif any(marker in lowered for marker in EVENT_MARKERS):
        query_kind = "event_lookup"
    else:
        query_kind = "general"

    company_terms: list[str] = []
    for match in COMPANY_PATTERN.finditer(text or ""):
        candidate = next((group for group in match.groups() if group), "")
        normalized = " ".join(candidate.split())
        if normalized:
            slug = _normalize_slug(normalized)
            if slug not in company_terms:
                company_terms.append(slug)

    search_terms = _tokenize_text(text)
    search_terms.extend(metric_names)
    search_terms.extend(topics)
    search_terms.extend(years)
    search_terms.extend(company_terms)

    deduped_search_terms: list[str] = []
    seen: set[str] = set()
    for item in search_terms:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_search_terms.append(normalized)

    diagnostics = [f"graph_interpret_query_kind={query_kind}"]
    if years:
        diagnostics.append(f"graph_interpret_years={','.join(years)}")
    return GraphQueryInterpretation(
        query_kind=query_kind,  # type: ignore[arg-type]
        metric_names=metric_names,
        topics=topics,
        years=years,
        company_terms=company_terms,
        search_terms=deduped_search_terms,
        comparison_mode=comparison_mode,
        diagnostics=diagnostics,
    )
