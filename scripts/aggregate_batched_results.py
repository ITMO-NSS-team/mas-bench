#!/usr/bin/env python3
"""Aggregate per-question metrics from results written by run_batched.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchlib.evaluation.llm_judge import judge_model, llm_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute full-benchmark metrics from batched result JSON files."
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <results_dir>/aggregate_results.json).",
    )
    parser.add_argument(
        "--judge-model",
        help="Model for semantic answer judging (default: JUDGE_MODEL or openai/gpt-4o-mini).",
    )
    return parser.parse_args()


def result_files(results_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in results_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if {"system_name", "benchmark", "model", "question_logs"} <= data.keys():
            files.append(path)
    return sorted(files)


def aggregate(data_files: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, data in data_files:
        groups[(data["system_name"], data["benchmark"], data["model"])].append(
            (path, data)
        )

    aggregates: list[dict[str, Any]] = []
    for (system_name, benchmark, model), runs in sorted(groups.items()):
        logs = [log for _, run in runs for log in run["question_logs"]]
        ids = [log["question_id"] for log in logs]
        duplicates = sorted({question_id for question_id in ids if ids.count(question_id) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate question ids for {system_name}/{benchmark}: {duplicates}"
            )
        if not logs:
            raise ValueError(f"No question logs for {system_name}/{benchmark}")

        metric_names = sorted({name for log in logs for name in log["metrics"]})
        metrics = {
            name: [log["metrics"][name] for log in logs if name in log["metrics"]]
            for name in metric_names
        }
        n = len(logs)
        aggregates.append(
            {
                "system_name": system_name,
                "benchmark": benchmark,
                "model": model,
                "total_questions": n,
                "failed_questions": sum(bool(log.get("error")) for log in logs),
                "avg_metrics": {
                    name: sum(scores) / len(scores) for name, scores in metrics.items()
                },
                "avg_metrics_successful_tasks": {
                    name: sum(scores) / len(scores) for name, scores in metrics.items()
                },
                "avg_metrics_all_tasks": {
                    name: sum(scores) / n for name, scores in metrics.items()
                },
                "avg_tokens_per_question": sum(log["total_tokens"] for log in logs) / n,
                "avg_prompt_tokens_per_question": (
                    sum(log["total_prompt_tokens"] for log in logs) / n
                ),
                "avg_completion_tokens_per_question": (
                    sum(log["total_completion_tokens"] for log in logs) / n
                ),
                "avg_tool_calls": sum(log["num_tool_calls"] for log in logs) / n,
                "avg_llm_calls": sum(log["num_llm_calls"] for log in logs) / n,
                "avg_latency_ms": sum(log["total_latency_ms"] for log in logs) / n,
                "source_files": [str(path) for path, _ in runs],
            }
        )
    return aggregates


def add_llm_accuracy(
    data_files: list[tuple[Path, dict[str, Any]]], model: str
) -> tuple[int, int]:
    """Score unjudged successful answers with the harness's original LLM judge."""
    logs = [
        log
        for _, data in data_files
        for log in data["question_logs"]
        if not log.get("error") and "llm_accuracy" not in log["metrics"]
    ]
    if logs:
        print(f"Scoring {len(logs)} answers with LLM judge ({model})...")
    scored = 0
    failed = 0
    for log in logs:
        usage = {"prompt": 0, "completion": 0}
        try:
            log["metrics"]["llm_accuracy"] = llm_accuracy(
                question=log["question"],
                predicted=log["predicted_answer"],
                gold=log["gold_answer"],
                model_name=model,
                usage_sink=usage,
            )
            log["judge_prompt_tokens"] = usage["prompt"]
            log["judge_completion_tokens"] = usage["completion"]
            scored += 1
        except Exception as exc:
            # This mirrors runner.py: a judge outage leaves the metric absent
            # for that question but does not discard the rest of the benchmark.
            failed += 1
            print(f"LLM judge failed for {log['question_id']}: {exc}")
    return scored, failed


def main() -> None:
    # Match benchlib.cli: the judge reads API credentials and JUDGE_MODEL from .env.
    load_dotenv()
    args = parse_args()
    paths = result_files(args.results_dir)
    if not paths:
        raise FileNotFoundError(f"No system result JSON files found under {args.results_dir}")
    files = [(path, json.loads(path.read_text())) for path in paths]

    selected_judge_model = args.judge_model or judge_model()
    scored, judge_failures = add_llm_accuracy(files, selected_judge_model)
    aggregates = aggregate(files)
    output = args.output or args.results_dir / "aggregate_results.json"
    output.write_text(
        json.dumps(
            {
                "judge_model": selected_judge_model,
                "newly_judged_questions": scored,
                "judge_failures": judge_failures,
                "aggregates": aggregates,
            },
            indent=2,
        )
        + "\n"
    )
    for result in aggregates:
        accuracy = result["avg_metrics_all_tasks"].get("llm_accuracy")
        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.4f}"
        print(
            f"{result['system_name']}/{result['benchmark']}: "
            f"questions={result['total_questions']}, failed={result['failed_questions']}, "
            f"llm_accuracy={accuracy_text}"
        )
    print(f"Saved aggregate results to {output}")


if __name__ == "__main__":
    main()
