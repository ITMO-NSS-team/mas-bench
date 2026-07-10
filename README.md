## Evaluating Auto-Generated Multi-Agent Systems on direct QA tasks

Fork of [mas-retrieval](../mas-retrieval) with the RAG setup removed: systems answer
questions directly (no retriever/corpus/index step), and only the systems that
auto-generate their own multi-agent structure are kept: **AutoMAS** and
**SwarmAgentic**.

`benchlib` is the harness (tracing, evaluation, CLI, contracts + discovery);
the content it measures is discovered by path:

```
experiments/
  systems/<name>/       # a system under test: __init__.py + adapter.py
  benchmarks/<name>/    # a benchmark: manifest.toml + builder.py + questions.jsonl
```

Bundled benchmarks: **seal_0** (111 q) and **seal_hard** (254 q) — subsets of
[SealQA](https://huggingface.co/datasets/vtllms/sealqa), long-horizon search QA;
**browsecomp** — deterministic 150-question subsample (seed 42) of
[BrowseComp](https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py),
multi-step web-browsing QA.

### Setup

```bash
# environment
uv sync                               # harness only
uv sync --group swarm_agentic         # + SwarmAgentic deps
uv pip install -e automas-research/   # + AutoMAS (vendored local source)

# SearXNG (shared web search backend for both systems; installs + starts in Docker,
# then set SEARXNG_URL=http://localhost:18888 in .env)
just searxng-start
just searxng-check    # verify the JSON API responds

# benchmark data (writes experiments/benchmarks/<name>/questions.jsonl)
uv sync --group benchmarks    # datasets, only needed for the seal_* downloads
just download seal_0
just download seal_hard
just download browsecomp      # decrypts locally; questions.jsonl is git-ignored, never commit it
```

`.env`: `OPENAI_API_KEY` / `OPENAI_BASE_URL` (AutoMAS also needs
`OPENROUTER_API_KEY`, or reuses `OPENAI_API_KEY` if `OPENAI_BASE_URL` points at
OpenRouter); `JUDGE_MODEL` overrides the fixed `llm_accuracy` judge (default
`openai/gpt-4o-mini`).

```bash
# list discovered benchmarks and systems
just available
```

### Run

```bash
# all flags: just run --help

# both systems on both SealQA subsets
just run --benchmark seal_0 seal_hard --systems automas swarm_agentic

# pick the worker model
just run --benchmark seal_0 --systems automas --model openai/gpt-4o

# mean ± std over repeats
just run --benchmark seal_0 --systems automas swarm_agentic --repeats 5

# split config: MAS construction on --meta-model, workers on --model
# (cross-vendor ids require OPENAI_BASE_URL -> OpenRouter)
just run --benchmark seal_0 seal_hard --systems automas swarm_agentic \
    --meta-model anthropic/claude-sonnet-4 --model openai/gpt-5-mini

# when the MAS is (re)generated: once per benchmark / fresh per question
just run --benchmark seal_0 --systems automas --generation-mode one_time
just run --benchmark seal_0 --systems automas --generation-mode per_task
```
