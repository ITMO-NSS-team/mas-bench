## Evaluating Auto-Generated Multi-Agent Systems on direct QA tasks

Fork of [mas-retrieval](../mas-retrieval) with the RAG setup removed: systems answer
questions directly (no retriever/corpus/index step), and only the systems that
auto-generate their own multi-agent structure are kept: **AutoMAS** and
**SwarmAgentic**.

Tiny library `benchlib` is the **harness**: tracing, evaluation, CLI, and the
adapter/benchmark contracts + discovery. The **content** it measures lives
outside the package and is discovered by path:

```
experiments/
  systems/<name>/       # a system under test: __init__.py + adapter.py
  benchmarks/<name>/    # a benchmark: manifest.toml + builder.py + questions.jsonl
```

Add a system or benchmark by dropping in a folder — no library edits.
`experiments/benchmarks/` is currently empty; non-RAG benchmarks (direct
question -> answer, no corpus) are added separately.

### Setup

```bash
uv sync                          # harness only
uv sync --group swarm_agentic    # + SwarmAgentic content deps
uv pip install -e automas-research/   # + AutoMAS (installs from vendored local source)
```

Set `OPENAI_API_KEY` / `OPENAI_BASE_URL` in `.env` (AutoMAS additionally needs
`OPENROUTER_API_KEY`, or reuses `OPENAI_API_KEY` if `OPENAI_BASE_URL` already
points at OpenRouter).

List what's available with `just available` (discovered benchmarks and systems).

### Run

Parameters are flags with defaults in `src/benchlib/cli.py` (`just run --help`).

```bash
just run --benchmark <name> --systems automas swarm_agentic
just run --benchmark <name> --systems automas --model openai/gpt-4o
just run --benchmark <name> --systems automas swarm_agentic --repeats 5
```

**Generation mode.** Both systems auto-generate their MAS and can do it once for
the whole benchmark or fresh for every question, pick with `--generation-mode`:

```bash
just run --benchmark <name> --systems automas --generation-mode one_time  # generate once, reuse across the benchmark
just run --benchmark <name> --systems automas --generation-mode per_task  # regenerate the MAS for each question
```

**Judge.** The `llm_accuracy` metric is scored by a fixed LLM judge
(`openai/gpt-4o-mini` by default, independent of `--model`); override with
`JUDGE_MODEL` in `.env`.
