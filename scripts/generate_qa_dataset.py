from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def resolve_output_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.parent.exists():
        return cwd_candidate
    return (ROOT_DIR / candidate).resolve()


def rollback_all_qa_source_states(
    doc_collection,
    *,
    from_state: int = 1,
    to_state: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not hasattr(doc_collection, "find") or not hasattr(doc_collection, "update_many"):
        raise RuntimeError("当前文档存储后端不支持全部回滚状态，请切换到支持 find/update_many 的实现。")

    query = {"state": from_state}
    matched_docs = list(doc_collection.find(query))
    matched_count = len(matched_docs)
    preview_node_ids = [doc.get("node_id") for doc in matched_docs[:20] if doc.get("node_id")]

    rolled_back_count = 0
    if not dry_run and matched_count > 0:
        result = doc_collection.update_many(query, {"$set": {"state": to_state}})
        rolled_back_count = int(getattr(result, "modified_count", matched_count) or 0)

    if matched_count == 0:
        message = "未找到需要回滚的文档"
    elif dry_run:
        message = "回滚预览完成，未实际修改状态"
    else:
        message = "全部 QA 源文档状态回滚完成"

    return {
        "mode": "rollback_all",
        "message": message,
        "success": True,
        "dry_run": dry_run,
        "from_state": from_state,
        "to_state": to_state,
        "matched_count": matched_count,
        "rolled_back_count": rolled_back_count,
        "preview_node_ids": preview_node_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA benchmark data from ingested RAG documents.")
    parser.add_argument(
        "--rollback-all",
        action="store_true",
        help="Rollback all source document states from one state to another.",
    )
    parser.add_argument("--source-state", type=int, default=2, help="Only process documents in this state.")
    parser.add_argument(
        "--mark-state",
        type=int,
        default=1,
        help="Update processed source documents to this state. Ignored when --dry-run is enabled.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of source documents to process.")
    parser.add_argument("--department-id", default=None, help="Only process documents for the given department.")
    parser.add_argument(
        "--dense-score-threshold",
        type=float,
        default=0.8,
        help="Minimum dense retrieval score for related context documents.",
    )
    parser.add_argument("--dense-top-k", type=int, default=5, help="Dense retrieval candidate count per source doc.")
    parser.add_argument("--max-related-docs", type=int, default=3, help="Maximum related docs kept per source doc.")
    parser.add_argument("--dry-run", action="store_true", help="Preview action result without writing changes.")
    parser.add_argument(
        "--rollback-from-state",
        type=int,
        default=1,
        help="Rollback all documents currently in this state.",
    )
    parser.add_argument(
        "--rollback-to-state",
        type=int,
        default=2,
        help="Rollback all matched documents to this state.",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default=None,
        help="Optional JSON export path for generated QA rows or rollback summary.",
    )
    return parser.parse_args()


def run_generate_mode(args: argparse.Namespace) -> dict[str, Any]:
    from src.rag.rag_service import rag_service

    return rag_service.generate_qa_dataset(
        source_state=args.source_state,
        mark_source_state=args.mark_state,
        limit=args.limit,
        department_id=args.department_id,
        dense_score_threshold=args.dense_score_threshold,
        dense_top_k=args.dense_top_k,
        max_related_docs=args.max_related_docs,
        dry_run=args.dry_run,
    )


def run_rollback_all_mode(args: argparse.Namespace) -> dict[str, Any]:
    from core.settings import settings
    from src.database.mongodb import mongodb_client

    doc_collection = mongodb_client.get_collection(settings.doc_collection_name)
    return rollback_all_qa_source_states(
        doc_collection,
        from_state=args.rollback_from_state,
        to_state=args.rollback_to_state,
        dry_run=args.dry_run,
    )


def write_export_payload(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if not args.export_path:
        return

    output_path = resolve_output_path(args.export_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rollback_all:
        export_payload = dict(summary)
    else:
        export_payload = summary.get("generated_rows") or []

    output_path.write_text(json.dumps(export_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["export_path"] = str(output_path)


def print_summary(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    printable_summary = dict(summary)
    if not args.rollback_all:
        printable_summary.pop("generated_rows", None)
    print(json.dumps(printable_summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    summary = run_rollback_all_mode(args) if args.rollback_all else run_generate_mode(args)
    write_export_payload(args, summary)
    print_summary(args, summary)


if __name__ == "__main__":
    main()
