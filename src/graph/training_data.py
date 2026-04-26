from __future__ import annotations

import json
from typing import Any


FACT_EXTRACTION_SYSTEM_PROMPT = """
你是一名财务报告事实提取专家。

仅提取所提供的文本块中明确支持的事实。

要求：
1. 仅返回 JSON。
2. 将事实分组在 `facts` 字段下。
3. 当文本块支持时，每条事实应使用以下可用架构字段：
   fact_kind, company_name, section_title, topic, metric_name, raw_value,
   numeric_value, unit, currency, period_end, period_type, summary, confidence。
4. fact_kind 请从以下选项中选择：
   - metric（指标）
   - event（事件）
   - risk（风险）
   - related_party（关联方）
   - policy（政策）
   - management_view（管理层观点）
5. 不要推断文本块中未明确陈述的值。
""".strip()


def _fact_get(fact: Any, field_name: str, default=None):
    if isinstance(fact, dict):
        return fact.get(field_name, default)
    return getattr(fact, field_name, default)


def serialize_fact_for_lora(fact: Any) -> dict:
    return {
        "fact_kind": _fact_get(fact, "fact_kind"),
        "company_name": _fact_get(fact, "company_name"),
        "section_title": _fact_get(fact, "section_title"),
        "topic": _fact_get(fact, "topic"),
        "metric_name": _fact_get(fact, "metric_name"),
        "raw_value": _fact_get(fact, "raw_value"),
        "numeric_value": _fact_get(fact, "numeric_value"),
        "unit": _fact_get(fact, "unit"),
        "currency": _fact_get(fact, "currency"),
        "period_end": _fact_get(fact, "period_end"),
        "period_type": _fact_get(fact, "period_type"),
        "summary": _fact_get(fact, "summary"),
        "confidence": _fact_get(fact, "confidence"),
    }


def build_fact_lora_example(node_id: str, evidence_doc: dict, facts: list[Any]) -> dict:
    metadata = evidence_doc.get("metadata") or {}
    user_prompt = "\n".join(
        [
            "Extract structured financial facts from the following report chunk.",
            f"node_id: {node_id}",
            f"file_name: {metadata.get('file_name', '')}",
            f"section_title: {metadata.get('section_title', '')}",
            f"page: {metadata.get('page', '')}",
            f"department_name: {metadata.get('department_name', '')}",
            "",
            "[Chunk]",
            evidence_doc.get("content", ""),
            "",
            "Return JSON only.",
            '{ "facts": [ ... ] }',
        ]
    )
    assistant_payload = {
        "facts": [serialize_fact_for_lora(fact) for fact in facts],
    }
    return {
        "node_id": node_id,
        "messages": [
            {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
        ],
    }


# Backward-compatible aliases for older script names and notebooks.
serialize_fact_for_sft = serialize_fact_for_lora
build_fact_sft_example = build_fact_lora_example
