from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.custom_types import DocumentMetadata
from src.graph.extractor import FinancialFactExtractor
from src.graph.training_data import build_fact_lora_example


TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv"}
JSON_SUFFIXES = {".json", ".jsonl"}

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


def _read_pdf_text(path: Path, *, max_pages: int, max_chars: int) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires PyMuPDF. Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    document = fitz.open(path)
    try:
        texts: list[str] = []
        page_limit = min(max_pages, len(document)) if max_pages > 0 else len(document)
        for page_index in range(page_limit):
            page = document.load_page(page_index)
            texts.append(page.get_text("text"))
            if max_chars > 0 and sum(len(item) for item in texts) >= max_chars:
                break
        joined = "\n".join(texts)
        return joined[:max_chars] if max_chars > 0 else joined
    finally:
        document.close()


def _read_text_file(path: Path, *, max_chars: int) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            text = raw.decode(encoding)
            return text[:max_chars] if max_chars > 0 else text
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, f"Unable to decode {path}")


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


def _extract_text_from_json_payload(payload) -> str:
    if isinstance(payload, dict):
        for field_name in TEXT_FIELD_CANDIDATES:
            text = _iter_nested_strings(payload.get(field_name))
            if text:
                return text
    return _iter_nested_strings(payload)


def _read_json_text(path: Path, *, max_chars: int) -> str:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(_extract_text_from_json_payload(json.loads(line)))
        text = "\n\n".join(item for item in rows if item)
        return text[:max_chars] if max_chars > 0 else text

    payload = json.loads(path.read_text(encoding="utf-8"))
    text = _extract_text_from_json_payload(payload)
    return text[:max_chars] if max_chars > 0 else text


def _read_source_text(path: Path, *, max_pages: int, max_chars: int) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_text(path, max_pages=max_pages, max_chars=max_chars)
    if suffix in TEXT_SUFFIXES:
        return _read_text_file(path, max_chars=max_chars)
    if suffix in JSON_SUFFIXES:
        return _read_json_text(path, max_chars=max_chars)
    return ""


def _load_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = {}
        for row in reader:
            file_name = (row.get("file_name") or "").strip()
            if file_name:
                rows[file_name] = {str(key): str(value or "") for key, value in row.items()}
        return rows


def _discover_input_files(input_dir: Path, *, patterns: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        for path in matches:
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            files.append(path)
    return sorted(files, key=lambda item: item.as_posix())


def _build_metadata(
    path: Path,
    *,
    input_dir: Path,
    manifest_row: dict[str, str],
    chunk_index: int,
    department_name: str,
    department_id: int,
) -> DocumentMetadata:
    try:
        relative_path = path.relative_to(input_dir).as_posix()
    except ValueError:
        relative_path = path.as_posix()

    section_title = (
        manifest_row.get("title")
        or manifest_row.get("category")
        or path.stem
    )

    return DocumentMetadata(
        file_name=path.name,
        file_path=relative_path,
        file_type=path.suffix.lower().lstrip("."),
        file_size=path.stat().st_size,
        source="local:data",
        section_title=section_title,
        page=0,
        chunk_index=chunk_index,
        department_id=department_id,
        department_name=department_name,
    )


def _node_id_for(path: Path, *, input_dir: Path, chunk_index: int) -> str:
    try:
        relative_path = path.relative_to(input_dir).as_posix()
    except ValueError:
        relative_path = path.as_posix()
    normalized = re.sub(r"[^0-9A-Za-z._/\-\u4e00-\u9fff]+", "_", relative_path).strip("_")
    return f"data::{normalized}::{chunk_index}"


def _positive_or_none(value: int) -> int | None:
    return value if value > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare financial fact LoRA training data from local files under the data directory."
    )
    parser.add_argument("--input-dir", default="data/chinese_documents_seed", help="Local input directory")
    parser.add_argument("--manifest", default=None, help="Optional CSV manifest path")
    parser.add_argument(
        "--patterns",
        default="*.pdf",
        help="Comma-separated file glob patterns, for example: *.pdf,*.txt,*.jsonl",
    )
    parser.add_argument("--recursive", action="store_true", help="Read matching files recursively")
    parser.add_argument("--output", default="data/financial_fact_lora_from_data.jsonl", help="Output JSONL path")
    parser.add_argument("--max-documents", type=int, default=100, help="Maximum source files to read; <=0 means all")
    parser.add_argument("--max-pages", type=int, default=8, help="Maximum PDF pages to extract per document")
    parser.add_argument("--max-chars", type=int, default=24000, help="Maximum characters to read per document")
    parser.add_argument("--chunk-size", type=int, default=1800, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--min-chars", type=int, default=300, help="Minimum chunk length")
    parser.add_argument("--max-facts-per-chunk", type=int, default=8, help="Maximum facts kept per chunk")
    parser.add_argument("--department-id", type=int, default=0, help="Metadata department id")
    parser.add_argument("--department-name", default="local_financial_reports", help="Metadata department name")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    manifest_path = Path(args.manifest) if args.manifest else input_dir / "manifest.csv"
    manifest_rows = _load_manifest(manifest_path)
    patterns = [item.strip() for item in args.patterns.split(",") if item.strip()]
    input_files = _discover_input_files(input_dir, patterns=patterns, recursive=args.recursive)
    max_documents = _positive_or_none(args.max_documents)
    if max_documents is not None:
        input_files = input_files[:max_documents]

    extractor = FinancialFactExtractor()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    documents_seen = 0
    examples_written = 0
    skipped_no_text = 0
    skipped_no_chunks = 0
    skipped_no_facts = 0
    errors: list[dict[str, str]] = []

    with output_path.open("w", encoding="utf-8") as fh:
        for path in input_files:
            try:
                text = _read_source_text(path, max_pages=args.max_pages, max_chars=args.max_chars)
            except Exception as exc:
                errors.append({"file": path.as_posix(), "error": str(exc)})
                continue

            if not text.strip():
                skipped_no_text += 1
                continue

            documents_seen += 1
            chunks = _chunk_text(
                text,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
            )
            if not chunks:
                skipped_no_chunks += 1
                continue

            manifest_row = manifest_rows.get(path.name, {})
            for chunk_index, chunk_text in enumerate(chunks):
                metadata = _build_metadata(
                    path,
                    input_dir=input_dir,
                    manifest_row=manifest_row,
                    chunk_index=chunk_index,
                    department_name=args.department_name,
                    department_id=args.department_id,
                )
                node_id = _node_id_for(path, input_dir=input_dir, chunk_index=chunk_index)
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
                "input_dir": str(input_dir),
                "manifest": str(manifest_path) if manifest_path.is_file() else None,
                "output": str(output_path),
                "files_found": len(input_files),
                "documents_seen": documents_seen,
                "examples_written": examples_written,
                "skipped_no_text": skipped_no_text,
                "skipped_no_chunks": skipped_no_chunks,
                "skipped_no_facts": skipped_no_facts,
                "errors": errors,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
