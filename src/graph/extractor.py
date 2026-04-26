from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

import jieba
from dateutil import parser as date_parser

from core.custom_types import DocumentMetadata
from core.settings import settings
from src.types.graph_type import (
    FinancialFact,
    GraphEntity,
    GraphEvidence,
    GraphExtractionBundle,
)


METRIC_ALIASES: dict[str, list[str]] = {
    "revenue": ["revenue", "total revenue", "营业收入", "收入"],
    "gross_written_premiums": ["gross written premiums", "保费收入"],
    "net_written_premiums": ["net written premiums", "净保费"],
    "profit_before_tax": ["profit before tax", "利润总额", "税前利润"],
    "profit_for_period": ["profit for the period", "net profit", "period profit", "本期利润", "净利润"],
    "total_assets": ["total assets", "资产总计", "总资产"],
    "total_liabilities": ["total liabilities", "负债总计", "总负债"],
    "equity": ["total equity", "equity", "所有者权益", "股东权益"],
    "cash_and_cash_equivalents": ["cash and cash equivalents", "现金及现金等价物"],
    "earnings_per_share": ["earnings per share", "eps", "每股收益"],
    "dividend_paid": ["dividend paid", "dividend", "股利", "分红"],
    "insurance_premium_receivable": ["insurance premium receivable", "保费应收"],
    "investments": ["investments", "投资"],
    "commission_receivable": ["commission receivable", "佣金应收"],
    "interest_income": ["interest income", "利息收入"],
    "other_operating_income": ["other operating income", "其他营业收入"],
}

SECTION_TOPICS: dict[str, list[str]] = {
    "income_statement": ["income statement", "statement of profit or loss", "损益表", "利润表"],
    "statement_of_financial_position": [
        "statement of financial position",
        "balance sheet",
        "财务状况表",
        "资产负债表",
    ],
    "cash_flow": ["cash flow", "现金流量表"],
    "capital_commitment": ["capital commitment", "capital commitments", "资本承诺"],
    "contingency": ["contingency", "contingencies", "或有负债", "或有事项"],
    "related_party": ["related party", "related party disclosures", "关联方", "关联交易"],
    "risk_management": ["risk management", "risk", "风险管理", "风险提示"],
    "acquisition": ["acquired", "acquisition", "purchase consideration", "收购", "并购"],
    "tax_assessment": ["tax", "assessment", "appeal", "vat", "nbt", "cgir", "tac", "税务", "评税", "上诉"],
    "regulatory_update": ["slfrs", "ifrs", "lkas", "regulation", "监管", "会计准则"],
}

EVENT_PATTERNS: list[tuple[str, list[str], str]] = [
    ("acquisition", ["acquired", "acquisition", "purchase consideration", "subsidiary"], "event"),
    ("capital_commitment", ["capital commitment", "capital commitments"], "event"),
    ("contingency", ["contingency", "contingencies"], "risk"),
    ("tax_assessment", ["assessment", "appeal", "cgir", "tac", "court of appeal"], "risk"),
    ("related_party", ["related party", "parent", "subsidiary", "key management personnel"], "related_party"),
    ("regulatory_update", ["slfrs", "ifrs", "lkas", "regulation", "compliance"], "policy"),
]

COMPANY_SUFFIXES = ("PLC", "LIMITED", "LTD", "INC", "CORP", "HOLDINGS", "COMPANY")
LINE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)+")
NUMBER_PATTERN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
DATE_CANDIDATE_PATTERN = re.compile(
    r"(?:(?:for the period ended|as at|ended|period ended|quarter ended)\s+)?"
    r"(\d{1,2}(?:st|nd|rd|th)?[\s./-]+[A-Za-z]+[\s./-]+\d{4}|\d{1,2}[\s./-]\d{1,2}[\s./-]\d{2,4})",
    re.IGNORECASE,
)


def _normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_slug(text: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip().lower()).strip("_")
    return normalized or "unknown"


def _stable_id(*parts: str) -> str:
    raw = "|".join(_normalize_space(part) for part in parts if part is not None)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text or "")


def _tokenize_text(text: str) -> list[str]:
    normalized = _normalize_space(text).lower()
    if not normalized:
        return []

    tokens: list[str] = []
    if _is_chinese(normalized):
        tokens.extend(token for token in jieba.lcut(normalized) if token.strip())
    tokens.extend(re.findall(r"[a-z0-9_]{2,}", normalized))

    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        item = token.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _sanitize_date_text(text: str) -> str:
    cleaned = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace(".", " ").replace("/", " ").replace("-", " ")
    return _normalize_space(cleaned)


def _parse_period_end(text: str) -> str | None:
    for match in DATE_CANDIDATE_PATTERN.finditer(text or ""):
        candidate = _sanitize_date_text(match.group(1))
        try:
            parsed = date_parser.parse(candidate, fuzzy=True, dayfirst=True)
            return parsed.date().isoformat()
        except (ValueError, OverflowError):
            continue
    return None


def _infer_period_type(text: str, file_name: str) -> str:
    normalized = f"{text} {file_name}".lower()
    if "annual report" in normalized or "integrated annual report" in normalized:
        return "annual"
    if "quarter" in normalized or "quarterly" in normalized:
        return "quarterly"
    if "interim" in normalized:
        return "interim"
    return "periodic"


def _infer_currency_and_unit(text: str) -> tuple[str | None, str | None]:
    lowered = (text or "").lower()
    currency = None
    if "rs" in lowered:
        currency = "LKR"
    elif "$" in lowered or "usd" in lowered:
        currency = "USD"

    unit = None
    if "million" in lowered:
        unit = "million"
    elif "billion" in lowered:
        unit = "billion"
    elif "000" in lowered or "' 000" in lowered or "'000" in lowered:
        unit = "thousand"
    return currency, unit


def _parse_numeric_value(raw_value: str | None) -> float | None:
    if not raw_value:
        return None
    candidate = raw_value.replace(",", "").strip()
    is_negative = candidate.startswith("(") and candidate.endswith(")")
    candidate = candidate.strip("()")
    try:
        value = float(candidate)
    except ValueError:
        return None
    return -value if is_negative else value


def _extract_company_name(text: str, metadata: DocumentMetadata) -> str:
    header = "\n".join((text or "").splitlines()[:8])
    for line in header.splitlines():
        cleaned = _normalize_space(line)
        if not cleaned:
            continue
        upper_line = cleaned.upper()
        if any(suffix in upper_line for suffix in COMPANY_SUFFIXES) and len(cleaned) <= 120:
            return cleaned

    section_hint = _normalize_space(metadata.file_name.rsplit(".", 1)[0])
    if section_hint:
        return section_hint
    return metadata.department_name or "Unknown Company"


def _extract_heading_title(text: str) -> str:
    for line in (text or "").splitlines()[:6]:
        cleaned = _normalize_space(line)
        if not cleaned:
            continue
        if len(cleaned) <= 120:
            return cleaned
    return ""


def _normalize_metric_name(line: str) -> str | None:
    lowered = (line or "").lower()
    for metric_name, aliases in METRIC_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return metric_name
    return None


def _normalize_topic(text: str) -> str:
    lowered = (text or "").lower()
    for topic, aliases in SECTION_TOPICS.items():
        if any(alias in lowered for alias in aliases):
            return topic
    return "general_financial_disclosure"


def _iter_candidate_lines(text: str) -> Iterable[str]:
    for raw_line in LINE_SPLIT_PATTERN.split(text or ""):
        cleaned = _normalize_space(raw_line.replace("|", " "))
        if len(cleaned) < 8:
            continue
        yield cleaned


class FinancialFactExtractor:
    """规则优先的财报事实抽取器。

    优先保证稳定产出结构化事实，后续可以把 extract_chunk 替换成
    微调后的抽取模型实现，而不用改存储层和 Graph-RAG 查询层。
    """

    def extract_chunk(
        self,
        *,
        node_id: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> GraphExtractionBundle:
        normalized_text = _normalize_space(text)
        if not node_id or not normalized_text:
            return GraphExtractionBundle(diagnostics=["graph_extract_empty_chunk"])

        company_name = _extract_company_name(text, metadata)
        period_end = _parse_period_end(f"{metadata.file_name} {text}") or None
        period_type = _infer_period_type(text, metadata.file_name)
        section_title = metadata.section_title or _extract_heading_title(text)
        topic = _normalize_topic(f"{section_title} {text[:300]}")
        report_id = _stable_id("report", metadata.file_path or metadata.file_name)
        currency, unit = _infer_currency_and_unit(text)
        evidence = GraphEvidence(node_id=node_id, content=text, metadata=metadata)

        metric_facts = self._extract_metric_facts(
            node_id=node_id,
            text=text,
            metadata=metadata,
            evidence=evidence,
            report_id=report_id,
            company_name=company_name,
            section_title=section_title,
            topic=topic,
            period_end=period_end,
            period_type=period_type,
            currency=currency,
            unit=unit,
        )
        event_facts = self._extract_event_facts(
            node_id=node_id,
            text=text,
            metadata=metadata,
            evidence=evidence,
            report_id=report_id,
            company_name=company_name,
            section_title=section_title,
            topic=topic,
            period_end=period_end,
            period_type=period_type,
        )

        facts = self._dedupe_facts([*metric_facts, *event_facts])[: settings.graph_max_facts_per_chunk]
        entities = self._build_entities(
            company_name=company_name,
            report_id=report_id,
            report_name=metadata.file_name,
            section_title=section_title,
            topic=topic,
            facts=facts,
            department_id=metadata.department_id,
        )
        diagnostics = [
            f"graph_company={company_name}",
            f"graph_topic={topic}",
            f"graph_metric_fact_count={len(metric_facts)}",
            f"graph_event_fact_count={len(event_facts)}",
        ]
        if period_end:
            diagnostics.append(f"graph_period_end={period_end}")
        return GraphExtractionBundle(entities=entities, facts=facts, diagnostics=diagnostics)

    def _extract_metric_facts(
        self,
        *,
        node_id: str,
        text: str,
        metadata: DocumentMetadata,
        evidence: GraphEvidence,
        report_id: str,
        company_name: str,
        section_title: str,
        topic: str,
        period_end: str | None,
        period_type: str,
        currency: str | None,
        unit: str | None,
    ) -> list[FinancialFact]:
        facts: list[FinancialFact] = []
        period_year = period_end[:4] if period_end else None

        for line in _iter_candidate_lines(text):
            metric_name = _normalize_metric_name(line)
            if not metric_name:
                continue
            raw_values = NUMBER_PATTERN.findall(line)
            raw_value = raw_values[0] if raw_values else None
            numeric_value = _parse_numeric_value(raw_value)
            aliases = METRIC_ALIASES.get(metric_name, [])
            search_terms = _tokenize_text(
                " ".join(
                    [
                        company_name,
                        section_title,
                        topic,
                        metric_name,
                        " ".join(aliases),
                        period_year or "",
                        line,
                    ]
                )
            )
            fact_key = _stable_id("metric", company_name, report_id, metric_name, line, node_id)
            facts.append(
                FinancialFact(
                    fact_id=fact_key,
                    fact_key=fact_key,
                    fact_kind="metric",
                    company_name=company_name,
                    normalized_company_name=_normalize_slug(company_name),
                    report_id=report_id,
                    report_name=metadata.file_name,
                    section_title=section_title,
                    topic=topic,
                    metric_name=metric_name,
                    normalized_metric_name=metric_name,
                    summary=line,
                    raw_value=raw_value,
                    numeric_value=numeric_value,
                    unit=unit,
                    currency=currency,
                    period_end=period_end,
                    period_year=period_year,
                    period_type=period_type,
                    confidence=0.78 if raw_value else 0.62,
                    department_id=metadata.department_id,
                    evidence_node_ids=[node_id],
                    evidence_docs=[evidence],
                    metadata={
                        "file_name": metadata.file_name,
                        "file_path": metadata.file_path,
                        "page": metadata.page,
                        "chunk_index": metadata.chunk_index,
                    },
                    search_terms=search_terms,
                    search_text=" ".join(search_terms),
                )
            )
        return facts

    def _extract_event_facts(
        self,
        *,
        node_id: str,
        text: str,
        metadata: DocumentMetadata,
        evidence: GraphEvidence,
        report_id: str,
        company_name: str,
        section_title: str,
        topic: str,
        period_end: str | None,
        period_type: str,
    ) -> list[FinancialFact]:
        facts: list[FinancialFact] = []
        period_year = period_end[:4] if period_end else None
        sentences = re.split(r"(?<=[。！？.!?;])\s+|\n", text or "")

        for sentence in sentences:
            normalized_sentence = _normalize_space(sentence)
            if len(normalized_sentence) < 20:
                continue

            lowered = normalized_sentence.lower()
            for event_topic, keywords, fact_kind in EVENT_PATTERNS:
                if not any(keyword in lowered for keyword in keywords):
                    continue

                counterparty_terms = re.findall(
                    r"[A-Z][A-Z&.,\-\s]{3,}(?:PLC|LIMITED|LTD|INC|CORP)",
                    normalized_sentence,
                )
                search_terms = _tokenize_text(
                    " ".join(
                        [
                            company_name,
                            event_topic,
                            section_title,
                            topic,
                            period_year or "",
                            " ".join(counterparty_terms),
                            normalized_sentence,
                        ]
                    )
                )
                fact_key = _stable_id(fact_kind, company_name, report_id, event_topic, normalized_sentence, node_id)
                facts.append(
                    FinancialFact(
                        fact_id=fact_key,
                        fact_key=fact_key,
                        fact_kind=fact_kind,  # type: ignore[arg-type]
                        company_name=company_name,
                        normalized_company_name=_normalize_slug(company_name),
                        report_id=report_id,
                        report_name=metadata.file_name,
                        section_title=section_title,
                        topic=event_topic or topic,
                        metric_name=None,
                        normalized_metric_name=None,
                        summary=normalized_sentence,
                        raw_value=NUMBER_PATTERN.findall(normalized_sentence)[0]
                        if NUMBER_PATTERN.findall(normalized_sentence)
                        else None,
                        numeric_value=_parse_numeric_value(
                            NUMBER_PATTERN.findall(normalized_sentence)[0]
                            if NUMBER_PATTERN.findall(normalized_sentence)
                            else None
                        ),
                        unit="million" if "million" in lowered else None,
                        currency="LKR" if "rs" in lowered else None,
                        period_end=period_end,
                        period_year=period_year,
                        period_type=period_type,
                        confidence=0.74,
                        department_id=metadata.department_id,
                        evidence_node_ids=[node_id],
                        evidence_docs=[evidence],
                        metadata={
                            "file_name": metadata.file_name,
                            "file_path": metadata.file_path,
                            "page": metadata.page,
                            "chunk_index": metadata.chunk_index,
                            "counterparties": counterparty_terms,
                        },
                        search_terms=search_terms,
                        search_text=" ".join(search_terms),
                    )
                )
                break

        return facts

    def _build_entities(
        self,
        *,
        company_name: str,
        report_id: str,
        report_name: str,
        section_title: str,
        topic: str,
        facts: list[FinancialFact],
        department_id: int,
    ) -> list[GraphEntity]:
        entity_map: dict[str, GraphEntity] = {}

        def add_entity(entity: GraphEntity) -> None:
            if entity.entity_id not in entity_map:
                entity_map[entity.entity_id] = entity

        add_entity(
            GraphEntity(
                entity_id=_stable_id("company", company_name),
                entity_type="company",
                name=company_name,
                normalized_name=_normalize_slug(company_name),
                department_id=department_id,
                metadata={"role": "primary_company"},
                search_terms=_tokenize_text(company_name),
            )
        )
        add_entity(
            GraphEntity(
                entity_id=report_id,
                entity_type="report",
                name=report_name,
                normalized_name=_normalize_slug(report_name),
                department_id=department_id,
                report_id=report_id,
                metadata={"company_name": company_name},
                search_terms=_tokenize_text(f"{report_name} {company_name}"),
            )
        )
        if section_title:
            add_entity(
                GraphEntity(
                    entity_id=_stable_id("section", report_id, section_title),
                    entity_type="section",
                    name=section_title,
                    normalized_name=_normalize_slug(section_title),
                    department_id=department_id,
                    report_id=report_id,
                    metadata={"topic": topic},
                    search_terms=_tokenize_text(section_title),
                )
            )
        if topic:
            add_entity(
                GraphEntity(
                    entity_id=_stable_id("topic", topic),
                    entity_type="topic",
                    name=topic,
                    normalized_name=_normalize_slug(topic),
                    department_id=department_id,
                    metadata={},
                    search_terms=_tokenize_text(topic),
                )
            )

        for fact in facts:
            if fact.normalized_metric_name:
                metric_name = fact.normalized_metric_name
                add_entity(
                    GraphEntity(
                        entity_id=_stable_id("metric", metric_name),
                        entity_type="metric",
                        name=metric_name,
                        normalized_name=_normalize_slug(metric_name),
                        department_id=department_id,
                        metadata={"topic": fact.topic},
                        search_terms=_tokenize_text(metric_name),
                    )
                )
            for counterparty in fact.metadata.get("counterparties", []) or []:
                add_entity(
                    GraphEntity(
                        entity_id=_stable_id("counterparty", counterparty),
                        entity_type="counterparty",
                        name=counterparty,
                        normalized_name=_normalize_slug(counterparty),
                        department_id=department_id,
                        metadata={},
                        search_terms=_tokenize_text(counterparty),
                    )
                )

        return list(entity_map.values())

    @staticmethod
    def _dedupe_facts(facts: list[FinancialFact]) -> list[FinancialFact]:
        deduped: dict[str, FinancialFact] = {}
        grouped_by_kind: defaultdict[str, int] = defaultdict(int)
        for fact in facts:
            if fact.fact_key not in deduped:
                grouped_by_kind[fact.fact_kind] += 1
                deduped[fact.fact_key] = fact

        ordered = list(deduped.values())
        ordered.sort(
            key=lambda item: (
                item.fact_kind != "metric",
                -(item.numeric_value is not None),
                -item.confidence,
            )
        )
        return ordered
