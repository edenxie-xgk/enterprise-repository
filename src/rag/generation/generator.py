import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from src.config.llm_config import LLMService
from src.prompts.rag.evidence_prompt import EVIDENCE_PROMPT

_NODE_ID_PATTERN = re.compile(r"\[node_id:([^\]\n]+)\]")
_CONTEXT_DOC_PATTERN = re.compile(r"\[node_id:([^\]\n]+)\]\n(.*?)(?=\n\n\[node_id:|\Z)", re.S)
_TOKEN_PATTERN = re.compile(
    "[A-Za-z]+(?:[-_][A-Za-z]+)*|[0-9][0-9,.\\-/%\\u5e74\\u6708\\u65e5]*|[\\u4e00-\\u9fff]{2,}"
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MAX_EVIDENCE_ATTEMPTS = 2


class EvidenceAssessmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_summary: str = Field(default="", description="Answer grounded in the provided evidence.")
    citations: list[str] = Field(default_factory=list, description="Referenced node IDs from the context.")
    is_sufficient: bool = Field(default=False, description="Whether the provided evidence is sufficient.")
    fail_reason: Literal[
        "low_recall",
        "bad_ranking",
        "ambiguous_query",
        "no_data",
        "insufficient_context",
    ] | None = Field(default=None, description="Failure reason when evidence is insufficient.")


def _normalize_answer_text(text: str) -> str:
    normalized = (text or "").replace("\r", " ").replace("\n", " ")
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    normalized = re.sub(r"\s*([，。！？；：])\s*", r"\1", normalized)
    normalized = re.sub(
        r"\s*,\s*",
        lambda match: "," if match.start() > 0
        and match.end() < len(normalized)
        and normalized[match.start() - 1].isdigit()
        and normalized[match.end()].isdigit()
        else ", ",
        normalized,
    )
    return normalized.strip()


def _extract_allowed_citations(context: str) -> list[str]:
    citations = []
    seen = set()
    for match in _NODE_ID_PATTERN.finditer(context or ""):
        node_id = match.group(1).strip()
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        citations.append(node_id)
    return citations


def _parse_context_docs(context: str) -> list[dict]:
    docs = []
    for match in _CONTEXT_DOC_PATTERN.finditer(context or ""):
        node_id = match.group(1).strip()
        content = (match.group(2) or "").strip()
        if not node_id or not content:
            continue
        docs.append({"node_id": node_id, "content": content})
    return docs


def _normalize_citations(raw_citations, allowed_citations: list[str]) -> list[str]:
    allowed_set = {item for item in allowed_citations if item}
    normalized = []
    seen = set()

    for item in raw_citations or []:
        citation = str(item).strip()
        if not citation or citation not in allowed_set or citation in seen:
            continue
        seen.add(citation)
        normalized.append(citation)

    return normalized


def _ensure_minimum_citations(
    citations: list[str],
    allowed_citations: list[str],
    *,
    min_citation_count: int,
) -> list[str]:
    target_count = max(0, min(int(min_citation_count or 0), len(allowed_citations)))
    if target_count <= len(citations):
        return citations

    seen = set(citations)
    supplemented = list(citations)
    for citation in allowed_citations:
        if citation in seen:
            continue
        seen.add(citation)
        supplemented.append(citation)
        if len(supplemented) >= target_count:
            break
    return supplemented


def _format_allowed_citations(allowed_citations: list[str]) -> str:
    if not allowed_citations:
        return "(none)"
    return "\n".join(f"- {citation}" for citation in allowed_citations)


def _extract_tokens(text: str) -> list[str]:
    return [item.strip().lower() for item in _TOKEN_PATTERN.findall(text or "") if item and item.strip()]


def _score_doc_for_citation(
    *,
    query: str,
    answer: str,
    doc_content: str,
    model_selected: bool,
) -> float:
    query_tokens = set(_extract_tokens(query))
    answer_tokens = set(_extract_tokens(answer))
    doc_tokens = set(_extract_tokens(doc_content))

    if not doc_tokens:
        return 0.0

    query_overlap = len(query_tokens & doc_tokens)
    answer_overlap = len(answer_tokens & doc_tokens)
    numeric_overlap = sum(
        1
        for token in answer_tokens
        if any(ch.isdigit() for ch in token) and token in doc_tokens
    )

    score = 0.0
    score += answer_overlap * 3.0
    score += query_overlap * 1.5
    score += numeric_overlap * 6.0
    if model_selected:
        score += 4.0
    return score


def _select_citations(
    *,
    query: str,
    answer: str,
    parsed_docs: list[dict],
    model_citations: list[str],
    min_citation_count: int,
) -> list[str]:
    allowed_citations = [doc["node_id"] for doc in parsed_docs if doc.get("node_id")]
    normalized = _normalize_citations(model_citations, allowed_citations)
    if not parsed_docs:
        return normalized

    if int(min_citation_count or 1) >= 2:
        target_count = len(parsed_docs)
    else:
        target_count = min(len(parsed_docs), 3)

    if target_count <= len(normalized):
        return normalized

    scored_docs = []
    selected_set = set(normalized)
    for index, doc in enumerate(parsed_docs):
        node_id = doc["node_id"]
        scored_docs.append(
            {
                "node_id": node_id,
                "score": _score_doc_for_citation(
                    query=query,
                    answer=answer,
                    doc_content=doc.get("content") or "",
                    model_selected=node_id in selected_set,
                ),
                "index": index,
            }
        )

    scored_docs.sort(key=lambda item: (-item["score"], item["index"]))
    citations = list(normalized)
    for item in scored_docs:
        node_id = item["node_id"]
        if node_id in selected_set:
            continue
        selected_set.add(node_id)
        citations.append(node_id)
        if len(citations) >= target_count:
            break
    return citations


def _normalize_response(
    response: EvidenceAssessmentResult | None,
    *,
    query: str,
    allowed_citations: list[str],
    parsed_docs: list[dict],
    min_citation_count: int,
) -> EvidenceAssessmentResult | None:
    if response is None:
        return None

    evidence_summary = _normalize_answer_text(response.evidence_summary or "")
    citations = _normalize_citations(response.citations, allowed_citations)

    if evidence_summary:
        citations = _select_citations(
            query=query,
            answer=evidence_summary,
            parsed_docs=parsed_docs,
            model_citations=citations,
            min_citation_count=min_citation_count,
        )

    if response.is_sufficient and evidence_summary:
        citations = _ensure_minimum_citations(
            citations,
            allowed_citations,
            min_citation_count=min_citation_count,
        )

    if evidence_summary == (response.evidence_summary or "") and citations == list(response.citations or []):
        return response

    return response.model_copy(
        update={
            "evidence_summary": evidence_summary,
            "citations": citations,
        }
    )


def _needs_retry(
    response: EvidenceAssessmentResult | None,
    *,
    min_citation_count: int,
    available_citation_count: int,
) -> bool:
    if response is None:
        return True

    if not (response.evidence_summary or "").strip():
        return True

    effective_min_citations = max(
        1,
        min(
            int(min_citation_count or 1),
            int(available_citation_count or 0) or int(min_citation_count or 1),
        ),
    )
    if response.is_sufficient and len(response.citations or []) < effective_min_citations:
        return True

    return False


def evaluate_evidence(
    llm: BaseChatModel,
    query: str,
    context: str,
    *,
    min_citation_count: int = 1,
) -> EvidenceAssessmentResult:
    allowed_citations = _extract_allowed_citations(context)
    parsed_docs = _parse_context_docs(context)
    prompt = EVIDENCE_PROMPT.format(
        query=query,
        context=context,
        allowed_citations=_format_allowed_citations(allowed_citations),
        min_citation_count=max(1, int(min_citation_count or 1)),
    )

    last_response = None
    for _ in range(_MAX_EVIDENCE_ATTEMPTS):
        response: EvidenceAssessmentResult = LLMService.invoke(
            llm=llm,
            messages=[HumanMessage(content=prompt)],
            schema=EvidenceAssessmentResult,
        )
        response = _normalize_response(
            response,
            query=query,
            allowed_citations=allowed_citations,
            parsed_docs=parsed_docs,
            min_citation_count=min_citation_count,
        )
        last_response = response
        if not _needs_retry(
            response,
            min_citation_count=min_citation_count,
            available_citation_count=len(allowed_citations),
        ):
            return response

    if last_response is None:
        return EvidenceAssessmentResult(
            evidence_summary="",
            citations=[],
            is_sufficient=False,
            fail_reason="insufficient_context",
        )

    if not (last_response.evidence_summary or "").strip():
        return EvidenceAssessmentResult(
            evidence_summary="",
            citations=list(last_response.citations or []),
            is_sufficient=False,
            fail_reason=last_response.fail_reason or "insufficient_context",
        )

    return last_response


generate_answer = evaluate_evidence
