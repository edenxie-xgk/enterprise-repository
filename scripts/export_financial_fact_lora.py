from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.settings import settings
from src.graph.store import FinancialGraphStore
from src.graph.training_data import build_fact_lora_example


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export financial fact extraction data for the LoRA training pipeline."
    )
    parser.add_argument("--output", default="data/financial_fact_lora.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=500, help="Maximum fact rows to read from the graph store")
    args = parser.parse_args()

    if not settings.graph_fact_collection_name:
        raise RuntimeError("GRAPH_FACT_COLLECTION_NAME is not configured")

    store = FinancialGraphStore()
    rows = list(store.facts.find({}, sort=[("updated_at", -1)], limit=max(1, args.limit)))
    grouped: dict[str, dict] = {}
    for row in rows:
        evidence_docs = row.get("evidence_docs") or []
        if not evidence_docs:
            continue
        evidence = evidence_docs[0]
        node_id = evidence.get("node_id") or ""
        if not node_id:
            continue
        if node_id not in grouped:
            grouped[node_id] = {
                "evidence": evidence,
                "facts": [],
            }
        grouped[node_id]["facts"].append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for node_id, payload in grouped.items():
            example = build_fact_lora_example(node_id, payload["evidence"], payload["facts"])
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Exported {len(grouped)} LoRA training examples to {output_path}")


if __name__ == "__main__":
    main()
