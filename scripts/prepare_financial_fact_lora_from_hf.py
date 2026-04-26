from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz

from core.custom_types import DocumentMetadata
from src.graph.extractor import FinancialFactExtractor
from src.graph.training_data import build_fact_lora_example


TEXT_FIELD_CANDIDATES = (
    "text",
    "content",
    "document_text",
    "report_text",
    "ocr_text",
    "plain_text",
    "markdown",
    "md_text",
)
FILE_FIELD_CANDIDATES = ("pdf", "document", "file")
NAME_FIELD_CANDIDATES = ("file_name", "filename", "name", "report_name", "title")
SECTION_FIELD_CANDIDATES = ("section_title", "section", "heading", "title")


def _import_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for LoRA dataset preparation from Hugging Face reports. "
            "Install it with `pip install datasets` or from requirements-train.txt."
        ) from exc
    return load_dataset


def _pick_first_string(row: dict, field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _iter_nested_strings(value) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for nested_value in value.values():
            result = _iter_nested_strings(nested_value)
            if result:
                return result
    if isinstance(value, list):
        for nested_value in value:
            result = _iter_nested_strings(nested_value)
            if result:
                return result
    return ""


def _extract_text_from_row(row: dict) -> str:
    for field_name in TEXT_FIELD_CANDIDATES:
        value = row.get(field_name)
        text = _iter_nested_strings(value)
        if text:
            return text
    return ""


def _read_pdf_text(pdf_value, *, max_pages: int, max_chars: int) -> str:
    pdf_bytes = None
    pdf_path = None

    if isinstance(pdf_value, bytes):
        pdf_bytes = pdf_value
    elif isinstance(pdf_value, str) and pdf_value.strip():
        pdf_path = pdf_value.strip()
    elif isinstance(pdf_value, dict):
        if isinstance(pdf_value.get("bytes"), bytes):
            pdf_bytes = pdf_value["bytes"]
        elif isinstance(pdf_value.get("path"), str) and pdf_value.get("path", "").strip():
            pdf_path = pdf_value["path"].strip()

    if pdf_bytes is None and not pdf_path:
        return ""

    if pdf_bytes is not None:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return ""
        document = fitz.open(pdf_file)

    try:
        texts: list[str] = []
        page_limit = min(max_pages, len(document)) if max_pages > 0 else len(document)
        for page_index in range(page_limit):
            page = document.load_page(page_index)
            texts.append(page.get_text("text"))
            if sum(len(item) for item in texts) >= max_chars:
                break
        return "\n".join(texts)[:max_chars]
    finally:
        document.close()


def _extract_pdf_text_from_row(row: dict, *, max_pages: int, max_chars: int) -> str:
    for field_name in FILE_FIELD_CANDIDATES:
        value = row.get(field_name)
        text = _read_pdf_text(value, max_pages=max_pages, max_chars=max_chars)
        if text:
            return text
    return ""


def _clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int, min_chars: int) -> list[str]:
    if not text:
        return []

    cleaned = _clean_text(text)
    if len(cleaned) < min_chars:
        return []

    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size].strip()
        if len(chunk) < min_chars:
            continue
        chunks.append(chunk)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


def _build_metadata(row: dict, *, file_name: str, chunk_index: int) -> DocumentMetadata:
    page = row.get("page")
    try:
        page_value = int(page) if page is not None else 0
    except (TypeError, ValueError):
        page_value = 0

    return DocumentMetadata(
        file_name=file_name,
        file_path=str(row.get("path") or row.get("file_path") or file_name),
        file_type="pdf",
        source="huggingface:ranzaka/cse_financial_reports",
        section_title=_pick_first_string(row, SECTION_FIELD_CANDIDATES),
        page=page_value,
        chunk_index=chunk_index,
        department_id=0,
        department_name="hf_financial_reports",
    )


def _row_file_name(row: dict, default_index: int) -> str:
    file_name = _pick_first_string(row, NAME_FIELD_CANDIDATES)
    if file_name:
        return file_name

    for field_name in FILE_FIELD_CANDIDATES:
        value = row.get(field_name)
        if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"].strip():
            return Path(value["path"]).name
        if isinstance(value, str) and value.strip():
            return Path(value).name
    return f"hf_report_{default_index}.pdf"


def _to_plain_row(row) -> dict:
    if isinstance(row, dict):
        return row
    if hasattr(row, "items"):
        return dict(row.items())
    return dict(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare financial fact LoRA training data from Hugging Face PDF reports."
    )
    parser.add_argument("--dataset", default="ranzaka/cse_financial_reports", help="Hugging Face dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output", default="data/financial_fact_lora_from_hf.jsonl", help="Output JSONL path")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache dir")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode to avoid full download")
    parser.add_argument("--max-documents", type=int, default=100, help="Maximum source rows to read")
    parser.add_argument("--max-pages", type=int, default=8, help="Maximum PDF pages to extract per document")
    parser.add_argument("--max-chars", type=int, default=24000, help="Maximum characters to read per document")
    parser.add_argument("--chunk-size", type=int, default=1800, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--min-chars", type=int, default=300, help="Minimum chunk length")
    parser.add_argument("--max-facts-per-chunk", type=int, default=8, help="Maximum facts kept per chunk")
    args = parser.parse_args()

    load_dataset = _import_datasets()
    extractor = FinancialFactExtractor()

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )
    if not args.streaming and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=42)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    documents_seen = 0
    examples_written = 0
    skipped_no_text = 0
    skipped_no_facts = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for row_index, row in enumerate(dataset):
            if documents_seen >= max(1, args.max_documents):
                break

            plain_row = _to_plain_row(row)
            file_name = _row_file_name(plain_row, row_index)
            text = _extract_text_from_row(plain_row)
            if not text:
                text = _extract_pdf_text_from_row(plain_row, max_pages=args.max_pages, max_chars=args.max_chars)
            if not text:
                skipped_no_text += 1
                continue

            documents_seen += 1
            chunks = _chunk_text(
                text,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
            )
            for chunk_index, chunk_text in enumerate(chunks):
                metadata = _build_metadata(plain_row, file_name=file_name, chunk_index=chunk_index)
                node_id = f"hf::{row_index}::{chunk_index}"
                bundle = extractor.extract_chunk(node_id=node_id, text=chunk_text, metadata=metadata)
                facts = bundle.facts[: max(1, args.max_facts_per_chunk)]
                if not facts:
                    skipped_no_facts += 1
                    continue

                evidence_doc = {
                    "content": chunk_text,
                    "metadata": metadata.model_dump(),
                }
                example = build_fact_lora_example(node_id, evidence_doc, facts)
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
                examples_written += 1

    print(
        json.dumps(
            {
                "output": str(output_path),
                "documents_seen": documents_seen,
                "examples_written": examples_written,
                "skipped_no_text": skipped_no_text,
                "skipped_no_facts": skipped_no_facts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
