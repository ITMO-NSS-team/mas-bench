#!/usr/bin/env python3
"""Run a benchmark in question batches, rotating the Tavily key per batch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark batches with a different Tavily key for each batch."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--tavily-keys-file",
        type=Path,
        required=True,
        help="One Tavily API key per line (blank lines and # comments are ignored).",
    )
    parser.add_argument(
        "--systems", nargs="+", default=["automas", "swarm_agentic"]
    )
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--meta-model")
    parser.add_argument("--generation-mode", choices=["one_time", "per_task"])
    parser.add_argument("--results-dir", type=Path, default=Path("results/batches"))
    parser.add_argument("--note", default="")
    return parser.parse_args()


def tavily_keys(path: Path) -> list[str]:
    keys = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys:
        raise ValueError(f"No Tavily keys found in {path}")
    return keys


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    source = Path("experiments/benchmarks") / args.benchmark
    manifest = source / "manifest.toml"
    questions_path = source / "questions.jsonl"
    if not manifest.is_file() or not questions_path.is_file():
        raise FileNotFoundError(f"Benchmark files not found under {source}")

    keys = tavily_keys(args.tavily_keys_file)
    questions = [json.loads(line) for line in questions_path.read_text().splitlines()]
    if not questions:
        raise ValueError(f"No questions found in {questions_path}")

    run_root = args.results_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{args.benchmark}"
    run_root.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="mas-bench-batches-") as temp_dir:
        data_dir = Path(temp_dir)
        batch_benchmark = data_dir / args.benchmark
        batch_benchmark.mkdir()
        (batch_benchmark / "manifest.toml").write_text(manifest.read_text())

        for batch_index, start in enumerate(range(0, len(questions), args.batch_size)):
            batch_questions = questions[start : start + args.batch_size]
            batch_dir = run_root / f"batch_{batch_index + 1:04d}"
            batch_dir.mkdir()
            (batch_dir / "batch.json").write_text(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "question_count": len(batch_questions),
                        "question_ids": [q.get("id") for q in batch_questions],
                        "status": "running",
                    },
                    indent=2,
                )
                + "\n"
            )
            (batch_benchmark / "questions.jsonl").write_text(
                "\n".join(json.dumps(q) for q in batch_questions) + "\n"
            )

            command = [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "benchlib.cli",
                "--benchmark",
                args.benchmark,
                "--systems",
                *args.systems,
                "--model",
                args.model,
                "--data-dir",
                str(data_dir),
                "--results-dir",
                str(batch_dir),
                "--note",
                f"{args.note} batch {batch_index + 1}".strip(),
            ]
            if args.meta_model:
                command.extend(["--meta-model", args.meta_model])
            if args.generation_mode:
                command.extend(["--generation-mode", args.generation_mode])

            # benchlib loads .env without overriding this value, so both AutoMAS
            # and SwarmAgentic use the selected key for this entire subprocess.
            env = os.environ.copy()
            env["TAVILY_API_KEY"] = keys[batch_index % len(keys)]
            print(
                f"Batch {batch_index + 1}: questions {start + 1}-{start + len(batch_questions)} "
                f"(Tavily key slot {batch_index % len(keys) + 1}/{len(keys)})"
            )
            completed = subprocess.run(command, env=env)

            status = "completed" if completed.returncode == 0 else "failed"
            (batch_dir / "batch.json").write_text(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "question_count": len(batch_questions),
                        "question_ids": [q.get("id") for q in batch_questions],
                        "status": status,
                        "exit_code": completed.returncode,
                    },
                    indent=2,
                )
                + "\n"
            )
            if completed.returncode:
                raise SystemExit(completed.returncode)

    print(f"Batch results saved under {run_root}")


if __name__ == "__main__":
    main()
