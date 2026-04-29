from __future__ import annotations

import json
import random
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config.llm_config import LLMService
from src.prompts.rag.qa_generation import QA_GENERATION_PROMPT
from utils.utils import get_current_time


_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


class QaData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., description="Question")
    answer: str = Field(..., description="Answer")
    language: str = Field(..., description="Language")
    difficulty: str = Field(..., description="Difficulty")
    intent: str = Field(..., description="Intent")
    node_ids: list[str] = Field(..., description="Supporting node ids")


class QAResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    qa_list: list[QaData] = Field(default_factory=list, description="Generated QA rows")


def _normalize_language_code(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("_", "-")
    if normalized in {"zh", "zh-cn", "zh-hans", "cn", "chinese"}:
        return "zh-cn"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return normalized or ""


def _detect_source_language(nodes: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    for key in ("language", "lang", "doc_language"):
        normalized = _normalize_language_code(metadata.get(key))
        if normalized:
            return normalized

    sample = "\n".join(str((node or {}).get("content") or "") for node in nodes[:3])
    cjk_count = len(_CJK_PATTERN.findall(sample))
    return "zh-cn" if cjk_count >= 8 else "en"


def _format_nodes(nodes: list[dict[str, Any]]) -> str:
    formatted = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        formatted.append(
            {
                "node_id": node.get("node_id"),
                "dense_score": node.get("dense_score"),
                "content": node.get("content"),
            }
        )
    return json.dumps(formatted, ensure_ascii=False, indent=2)


def generate_qa(
    llm: BaseChatModel,
    nodes: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    *,
    max_qa_per_batch: int = 2,
) -> list[dict[str, Any]]:
    source_language = _detect_source_language(nodes, metadata)
    prompt = QA_GENERATION_PROMPT.format(
        nodes=_format_nodes(nodes),
        source_language=source_language,
        max_qa_per_batch=max(1, max_qa_per_batch),
    )
    node_ids = {node["node_id"] for node in nodes if isinstance(node, dict) and node.get("node_id")}

    response: QAResult = LLMService.invoke(
        schema=QAResult,
        messages=[HumanMessage(content=prompt)],
        llm=llm,
    )

    qa_list: list[dict[str, Any]] = []
    create_time = get_current_time()
    for item in response.qa_list:
        question = (item.question or "").strip()
        answer = (item.answer or "").strip()
        item_language = _normalize_language_code(item.language) or source_language
        difficulty = (item.difficulty or "").strip().lower()
        intent = (item.intent or "").strip().lower()
        item_node_ids = [str(node_id).strip() for node_id in item.node_ids if str(node_id).strip()]

        if not question or not answer or not item_node_ids:
            continue
        if len(set(item_node_ids)) < 2:
            continue
        if any(node_id not in node_ids for node_id in item_node_ids):
            continue
        if item_language != source_language:
            continue
        if difficulty not in {"easy", "medium", "hard"}:
            continue
        if intent not in {"factoid", "comparison", "analysis"}:
            continue

        qa_list.append(
            {
                "question": question,
                "answer": answer,
                "language": item_language,
                "difficulty": difficulty,
                "intent": intent,
                "metadata": metadata,
                "node_ids": item_node_ids,
                "create_time": create_time,
            }
        )

    validation_index = random.randrange(len(qa_list)) if len(qa_list) > 1 else None
    for index, item in enumerate(qa_list):
        item["state"] = 2 if validation_index is not None and index == validation_index else 0

    return qa_list
