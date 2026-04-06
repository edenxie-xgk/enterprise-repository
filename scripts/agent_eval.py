from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.agent.runner import build_run_report, run_agent


DEFAULT_INPUT = Path("data/rag_agent.rag_qa.json")
DEFAULT_OUTPUT_JSON = Path("data/agent_eval/agent_eval_report_v2.json")
DEFAULT_OUTPUT_JSONL = Path("data/agent_eval/agent_eval_report_v2.jsonl")


def load_qa_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_report(
    report: dict[str, Any],
    *,
    gold_node_ids: list[str],
    retrieval_hit_count: int,
    rerank_hit_count: int,
) -> tuple[str, str]:
    answer = (report.get("answer") or "").strip()
    state_status = report.get("status")
    state_fail_reason = report.get("fail_reason")
    last_rag_result = report.get("last_rag_result") or {}
    rag_fail_reason = last_rag_result.get("fail_reason")
    rag_is_sufficient = bool(last_rag_result.get("is_sufficient"))
    citations = report.get("citations") or last_rag_result.get("citations") or []
    document_count = last_rag_result.get("document_count") or 0
    has_any_gold = bool(gold_node_ids)
    retrieval_has_any_gold_hit = retrieval_hit_count > 0
    rerank_has_any_gold_hit = rerank_hit_count > 0
    full_gold_coverage = has_any_gold and rerank_hit_count >= len(gold_node_ids)

    if (
        state_status == "success"
        and rag_is_sufficient
        and answer
        and citations
        and document_count > 0
        and (not has_any_gold or rerank_has_any_gold_hit)
    ):
        if full_gold_coverage:
            return "success", "final_answer_with_full_gold_evidence"
        return "success", "final_answer_with_sufficient_evidence"

    if answer and (citations or document_count > 0):
        if has_any_gold and retrieval_has_any_gold_hit and not rerank_has_any_gold_hit:
            return "partial_success", "gold_evidence_retrieved_but_lost_after_rerank_or_finalize"
        if has_any_gold and rerank_has_any_gold_hit and not full_gold_coverage:
            return "partial_success", "answer_available_with_partial_gold_coverage"
        return "partial_success", "answer_available_but_evidence_insufficient_or_incomplete"

    if answer and state_fail_reason in {"max_steps_exceeded", "insufficient_context"}:
        return "partial_success", "fallback_answer_after_incomplete_run"

    if has_any_gold and retrieval_has_any_gold_hit and not rerank_has_any_gold_hit:
        return "failed", "gold_evidence_dropped_after_rerank"

    if rag_fail_reason in {"bad_ranking", "low_recall", "no_data", "insufficient_context", "verification_failed"}:
        return "failed", f"retrieval_pipeline_failed:{rag_fail_reason}"

    if state_fail_reason:
        return "failed", f"agent_failed:{state_fail_reason}"

    return "failed", "no_answer_and_no_reliable_evidence"


def build_eval_record(sample: dict[str, Any], report: dict[str, Any], run_seconds: float) -> dict[str, Any]:
    last_rag_result = report.get("last_rag_result") or {}
    gold_node_ids = sample.get("node_ids") or []
    retrieval_candidate_node_ids = last_rag_result.get("retrieval_candidate_node_ids") or []
    rerank_node_ids = last_rag_result.get("rerank_node_ids") or []
    gold_node_id_set = set(gold_node_ids)
    retrieval_hit_node_ids = [node_id for node_id in retrieval_candidate_node_ids if node_id in gold_node_id_set]
    rerank_hit_node_ids = [node_id for node_id in rerank_node_ids if node_id in gold_node_id_set]
    outcome, outcome_reason = classify_report(
        report,
        gold_node_ids=gold_node_ids,
        retrieval_hit_count=len(retrieval_hit_node_ids),
        rerank_hit_count=len(rerank_hit_node_ids),
    )
    return {
        "index": sample.get("index"),
        "question": sample.get("question"),
        "intent": sample.get("intent"),
        "difficulty": sample.get("difficulty"),
        "language": sample.get("language"),
        "node_ids": sample.get("node_ids") or [],
        "run_seconds": round(run_seconds, 3),
        "outcome": outcome,
        "outcome_reason": outcome_reason,
        "agent_status": report.get("status"),
        "agent_fail_reason": report.get("fail_reason"),
        "rag_fail_reason": last_rag_result.get("fail_reason"),
        "rag_is_sufficient": last_rag_result.get("is_sufficient"),
        "citation_count": len(report.get("citations") or last_rag_result.get("citations") or []),
        "document_count": last_rag_result.get("document_count") or 0,
        "gold_node_ids": gold_node_ids,
        "retrieval_candidate_node_ids": retrieval_candidate_node_ids,
        "rerank_node_ids": rerank_node_ids,
        "retrieval_hit_node_ids": retrieval_hit_node_ids,
        "rerank_hit_node_ids": rerank_hit_node_ids,
        "retrieval_hit_count": len(retrieval_hit_node_ids),
        "rerank_hit_count": len(rerank_hit_node_ids),
        "retrieval_hit_ratio": round(len(retrieval_hit_node_ids) / len(gold_node_ids), 4) if gold_node_ids else 0.0,
        "rerank_hit_ratio": round(len(rerank_hit_node_ids) / len(gold_node_ids), 4) if gold_node_ids else 0.0,
        "retrieval_has_any_gold_hit": bool(retrieval_hit_node_ids),
        "rerank_has_any_gold_hit": bool(rerank_hit_node_ids),
        "answer": report.get("answer"),
        "evidence_summary": last_rag_result.get("evidence_summary"),
        "report": report,
        "expected_answer": sample.get("answer"),
    }


def run_single_sample(sample: dict[str, Any], max_steps: int) -> dict[str, Any]:
    start_time = time.time()
    state = run_agent(
        sample.get("question") or "",
        user_id=str((sample.get("metadata") or {}).get("user_id", "")),
        session_id=f"eval-{sample.get('index', 0)}",
        user_profile=sample.get("metadata") or {},
        max_steps=max_steps,
    )
    run_seconds = time.time() - start_time
    report = build_run_report(state)
    return build_eval_record(sample, report, run_seconds)


def run_eval(
    dataset: list[dict[str, Any]],
    *,
    max_steps: int,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    runtime_errors: list[dict[str, Any]] = []

    started_at = time.time()
    samples = dataset[:limit] if limit else dataset

    outcome_counter: Counter[str] = Counter()
    agent_status_counter: Counter[str] = Counter()
    agent_fail_reason_counter: Counter[str] = Counter()
    rag_fail_reason_counter: Counter[str] = Counter()
    retrieval_gold_hit_counter: Counter[str] = Counter()
    rerank_gold_hit_counter: Counter[str] = Counter()
    intent_outcome_counter: Counter[str] = Counter()
    difficulty_outcome_counter: Counter[str] = Counter()

    for index, sample in enumerate(samples, start=1):
        sample = dict(sample)
        sample.setdefault("index", index)
        try:
            record = run_single_sample(sample, max_steps=max_steps)
            results.append(record)

            outcome_counter.update([record["outcome"]])
            if record.get("agent_status"):
                agent_status_counter.update([record["agent_status"]])
            if record.get("agent_fail_reason"):
                agent_fail_reason_counter.update([record["agent_fail_reason"]])
            if record.get("rag_fail_reason"):
                rag_fail_reason_counter.update([record["rag_fail_reason"]])
            retrieval_gold_hit_counter.update(["hit" if record.get("retrieval_has_any_gold_hit") else "miss"])
            rerank_gold_hit_counter.update(["hit" if record.get("rerank_has_any_gold_hit") else "miss"])
            if record.get("intent"):
                intent_outcome_counter.update([f"{record['intent']}::{record['outcome']}"])
            if record.get("difficulty"):
                difficulty_outcome_counter.update([f"{record['difficulty']}::{record['outcome']}"])
        except Exception as exc:
            runtime_errors.append(
                {
                    "index": sample.get("index"),
                    "question": sample.get("question"),
                    "intent": sample.get("intent"),
                    "difficulty": sample.get("difficulty"),
                    "language": sample.get("language"),
                    "outcome": "error",
                    "error": str(exc),
                }
            )
            outcome_counter.update(["error"])

    summary = {
        "total": len(samples),
        "completed": len(results),
        "runtime_error_count": len(runtime_errors),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "outcome_counter": dict(outcome_counter),
        "agent_status_counter": dict(agent_status_counter),
        "agent_fail_reason_counter": dict(agent_fail_reason_counter),
        "rag_fail_reason_counter": dict(rag_fail_reason_counter),
        "retrieval_gold_hit_counter": dict(retrieval_gold_hit_counter),
        "rerank_gold_hit_counter": dict(rerank_gold_hit_counter),
        "intent_outcome_counter": dict(intent_outcome_counter),
        "difficulty_outcome_counter": dict(difficulty_outcome_counter),
    }
    return results, {"summary": summary, "runtime_errors": runtime_errors}


def write_outputs(
    results: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    output_json: Path,
    output_jsonl: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(
            {
                **meta,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def agent_eval() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agentic RAG with enterprise-friendly outcome buckets.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="QA dataset path")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output report JSON path")
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL), help="Output report JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples")
    parser.add_argument("--max-steps", type=int, default=6, help="Max agent steps")
    args = parser.parse_args()

    dataset = load_qa_dataset(Path(args.input))
    results, meta = run_eval(dataset, max_steps=args.max_steps, limit=args.limit)
    write_outputs(
        results,
        meta,
        output_json=Path(args.output_json),
        output_jsonl=Path(args.output_jsonl),
    )

    print(json.dumps(meta["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    agent_eval()
