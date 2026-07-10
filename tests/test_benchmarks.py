from __future__ import annotations

from benchlib.benchmarks.base import (
    DEFAULT_METRICS,
    _spec_from_manifest,
    discover,
    load_spec,
    slugify,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello_world"

    def test_collapses_non_alnum(self):
        assert slugify("Foo -- Bar!!Baz") == "foo_bar_baz"

    def test_strips_edges(self):
        assert slugify("  --Trim--  ") == "trim"


class TestSpecFromManifest:
    def test_full_manifest(self, tmp_path):
        bench = tmp_path / "mybench"
        bench.mkdir()
        (bench / "manifest.toml").write_text(
            'name = "mybench"\n'
            'description = "  A bench  "\n'
            'metrics = ["f1", "exact_match"]\n'
        )
        spec = _spec_from_manifest(bench / "manifest.toml")
        assert spec.name == "mybench"
        assert spec.description == "A bench"
        assert spec.metrics == ("f1", "exact_match")
        assert spec.questions_path == bench / "questions.jsonl"

    def test_defaults_when_absent(self, tmp_path):
        bench = tmp_path / "minimal"
        bench.mkdir()
        (bench / "manifest.toml").write_text("")
        spec = _spec_from_manifest(bench / "manifest.toml")
        # name falls back to the directory name.
        assert spec.name == "minimal"
        assert spec.metrics == DEFAULT_METRICS


class TestDiscover:
    def test_empty_dir(self, tmp_path):
        assert discover(tmp_path) == {}

    def test_missing_dir(self, tmp_path):
        assert discover(tmp_path / "nope") == {}

    def test_finds_manifest(self, tmp_path):
        bench = tmp_path / "b1"
        bench.mkdir()
        (bench / "manifest.toml").write_text('name = "b1"\n')
        specs = discover(tmp_path)
        assert set(specs) == {"b1"}
        assert load_spec("b1", tmp_path).name == "b1"

    def test_load_spec_unknown_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_spec("ghost", tmp_path)
