import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_batched.py"
SPEC = importlib.util.spec_from_file_location("run_batched", SCRIPT)
assert SPEC and SPEC.loader
run_batched = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_batched)


def test_display_key_masks_the_secret() -> None:
    assert run_batched.display_key("tvly-dev-abcdefghijklmnopwxzy") == "tvly-dev-abc...wxzy"
    assert run_batched.display_key("short-key") == "<redacted>"


def test_batches_checkpoint_questions_and_rotate_tavily_keys(
    tmp_path: Path, monkeypatch
) -> None:
    benchmark = tmp_path / "experiments/benchmarks/demo"
    benchmark.mkdir(parents=True)
    (benchmark / "manifest.toml").write_text('name = "demo"\n')
    (benchmark / "questions.jsonl").write_text(
        "".join(
            json.dumps({"id": str(i), "question": f"q{i}", "answer": f"a{i}"})
            + "\n"
            for i in range(5)
        )
    )
    keys = tmp_path / "keys"
    keys.write_text("first\nsecond\n")
    seen: list[tuple[str, list[str]]] = []

    def fake_run(command: list[str], env: dict[str, str]):
        data_dir = Path(command[command.index("--data-dir") + 1])
        seen.append(
            (
                env["TAVILY_API_KEY"],
                [json.loads(line)["id"] for line in (data_dir / "demo/questions.jsonl").read_text().splitlines()],
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_batched.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--benchmark",
            "demo",
            "--batch-size",
            "2",
            "--tavily-keys-file",
            str(keys),
            "--reverse-tavily-keys",
            "--results-dir",
            "batch-results",
        ],
    )

    run_batched.main()

    assert seen == [
        ("second", ["0", "1"]),
        ("first", ["2", "3"]),
        ("second", ["4"]),
    ]
    checkpoints = sorted((tmp_path / "batch-results").glob("*/batch_*/batch.json"))
    assert [json.loads(path.read_text())["status"] for path in checkpoints] == [
        "completed",
        "completed",
        "completed",
    ]
