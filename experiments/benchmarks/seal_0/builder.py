"""Builder for the SealQA Seal-0 subset (vtllms/sealqa, config ``seal_0``).

Writes ``questions.jsonl`` with the harness's ``{"id", "question", "answer"}``
contract plus SealQA's temporal metadata (``freshness`` / ``effective_year`` —
useful for filtering stale gold answers) and the contamination-detection
``canary`` string, which must travel with any copy of the data. The dataset's
retrieval fields (``urls``, ``search_results``) are dropped: in this no-RAG
setup systems gather evidence with their own tools.
"""

from __future__ import annotations

import json

from benchlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register

HF_DATASET = "vtllms/sealqa"
HF_CONFIG = "seal_0"


@register("seal_0")
class Seal0Builder(BenchmarkBuilder):
    def download(self, spec: BenchmarkSpec) -> None:
        from datasets import load_dataset

        rows = load_dataset(HF_DATASET, HF_CONFIG, split=spec.split or "test")
        with open(spec.questions_path, "w") as f:
            for i, row in enumerate(rows):
                record = {
                    "id": f"{HF_CONFIG}_{i:04d}",
                    "question": row["question"],
                    "answer": row["answer"],
                    "freshness": row.get("freshness", ""),
                    "effective_year": row.get("effective_year", ""),
                    "topic": row.get("topic", ""),
                    "question_types": row.get("question_types", []),
                    "canary": row.get("canary", ""),
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(rows)} questions to {spec.questions_path}")
