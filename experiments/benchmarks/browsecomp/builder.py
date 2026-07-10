"""Builder for a BrowseComp subsample (openai/simple-evals).

The official test set ships XOR-encrypted (key derived from the per-row
``canary``) precisely so plaintext never lands in training corpora;
``derive_key``/``decrypt`` are copied verbatim from
https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py.
The decrypted ``questions.jsonl`` therefore must never be committed or
published (git-ignored; rebuild locally with ``just download browsecomp``).

Writes a deterministic ``SAMPLE_N``-question subsample (seed ``SEED``) of the
1,266-question set; ids keep the original row index so runs stay traceable to
the full set. Stdlib-only — no extra dependency groups needed.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import random
import urllib.request

from benchlib.benchmarks.base import BenchmarkBuilder, BenchmarkSpec, register

CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
SAMPLE_N = 150
SEED = 42


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


@register("browsecomp")
class BrowseCompBuilder(BenchmarkBuilder):
    def download(self, spec: BenchmarkSpec) -> None:
        with urllib.request.urlopen(CSV_URL) as resp:
            text = resp.read().decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))

        indices = sorted(random.Random(SEED).sample(range(len(rows)), SAMPLE_N))
        with open(spec.questions_path, "w") as f:
            for i in indices:
                row = rows[i]
                canary = row.get("canary", "")
                record = {
                    "id": f"browsecomp_{i:04d}",
                    "question": decrypt(row.get("problem", ""), canary),
                    "answer": decrypt(row.get("answer", ""), canary),
                    "canary": canary,
                }
                f.write(json.dumps(record) + "\n")
        print(
            f"Wrote {len(indices)} of {len(rows)} questions to {spec.questions_path}"
        )
