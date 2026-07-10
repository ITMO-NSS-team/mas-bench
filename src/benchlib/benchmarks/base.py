from __future__ import annotations

import importlib
import json
import re
import sys
import tomllib
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchlib.log import logger

# Default location of the benchmark repository (one subdirectory per benchmark).
DEFAULT_ROOT = Path("experiments/benchmarks")
_BENCH_NS = "_benchlib_benchmarks"

# Metrics applied when a manifest does not declare its own ``metrics`` list.
DEFAULT_METRICS: tuple[str, ...] = ("exact_match", "f1")


def slugify(text: str) -> str:
    """Lowercase text and replace runs of non-alphanumeric chars with underscores."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


@dataclass(frozen=True)
class BenchmarkSpec:
    """A benchmark loaded from its ``manifest.toml``; paths are conventions under ``root``."""

    name: str
    root: Path
    description: str
    split: str | None = None
    metrics: tuple[str, ...] = DEFAULT_METRICS

    @property
    def questions_path(self) -> Path:
        """JSONL of evaluation questions, one ``{"id", "question", "answer", ...}`` per line."""
        return self.root / "questions.jsonl"

    def load_questions(self, sample_n: int | None = None) -> list[dict[str, Any]]:
        """Read the question set from ``questions_path`` (optionally capped)."""
        with open(self.questions_path) as f:
            questions = [json.loads(line) for line in f]
        if sample_n is not None:
            return questions[:sample_n]
        return questions


def _spec_from_manifest(manifest_path: Path) -> BenchmarkSpec:
    """Parse a single ``manifest.toml`` into a :class:`BenchmarkSpec`."""
    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)
    root = manifest_path.parent
    name = data.get("name") or root.name
    metrics = data.get("metrics")
    return BenchmarkSpec(
        name=name,
        root=root,
        description=(data.get("description") or "").strip(),
        split=data.get("split"),
        metrics=tuple(metrics) if metrics else DEFAULT_METRICS,
    )


def _ensure_namespace(parent: str, search_dir: Path) -> None:
    """Register/refresh a synthetic namespace package rooted at ``search_dir``."""
    mod = sys.modules.get(parent)
    if mod is None:
        mod = types.ModuleType(parent)
        sys.modules[parent] = mod
    mod.__path__ = [str(search_dir)]  # type: ignore[attr-defined]


def discover(root: str | Path = DEFAULT_ROOT) -> dict[str, BenchmarkSpec]:
    """Scan ``<root>/*/manifest.toml`` for benchmarks, importing each builder.py so
    its ``@register`` fires. The available set is exactly the dirs with a manifest."""
    root = Path(root)
    specs: dict[str, BenchmarkSpec] = {}
    if not root.is_dir():
        return specs
    _ensure_namespace(_BENCH_NS, root)
    for manifest_path in sorted(root.glob("*/manifest.toml")):
        spec = _spec_from_manifest(manifest_path)
        specs[spec.name] = spec
        if (manifest_path.parent / "builder.py").exists():
            try:
                importlib.import_module(f"{_BENCH_NS}.{spec.name}.builder")
            except Exception as e:  # builder import should be light; warn if not
                logger.warning(f"Builder for '{spec.name}' failed to load: {e!r}")
    return specs


def load_spec(name: str, root: str | Path = DEFAULT_ROOT) -> BenchmarkSpec:
    """Load one benchmark spec by name, raising with the available list on miss."""
    specs = discover(root)
    if name not in specs:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available in {root}: {sorted(specs)}"
        )
    return specs[name]


class BenchmarkBuilder(ABC):
    """Fetch and prepare one benchmark's data.

    Import heavy deps (``datasets``, ...) lazily inside methods so importing the
    registry stays dependency-free.
    """

    @abstractmethod
    def download(self, spec: BenchmarkSpec) -> None:
        """Fetch/build ``spec.questions_path`` (and any raw source files)."""


_BUILDERS: dict[str, type[BenchmarkBuilder]] = {}


def register(name: str):
    """Class decorator registering a :class:`BenchmarkBuilder` under ``name``."""

    def deco(cls: type[BenchmarkBuilder]) -> type[BenchmarkBuilder]:
        if name in _BUILDERS:
            raise ValueError(f"Benchmark builder '{name}' already registered")
        _BUILDERS[name] = cls
        return cls

    return deco


def registered_builders() -> list[str]:
    """Names of all registered builders (import the package to populate)."""
    return sorted(_BUILDERS)


def get_builder(name: str) -> BenchmarkBuilder:
    """Instantiate the registered builder for ``name``, raising with the list on miss."""
    if name not in _BUILDERS:
        raise ValueError(
            f"No builder registered for '{name}'. Registered: {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name]()
