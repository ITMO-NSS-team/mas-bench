from __future__ import annotations

import pytest

from benchlib.evaluation.base import EvalContext, get_metric
from benchlib.evaluation.metrics import exact_match, f1_score, normalize_answer


class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("Hello WORLD") == "hello world"

    def test_drops_articles(self):
        assert normalize_answer("the a cat an dog") == "cat dog"

    def test_strips_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_answer("  foo   bar\t baz ") == "foo bar baz"

    def test_article_only_inside_words_kept(self):
        # "the" as a substring of a token must not be stripped.
        assert normalize_answer("theory") == "theory"


class TestExactMatch:
    def test_match_after_normalization(self):
        assert exact_match("The Cat.", "a cat") == 1.0

    def test_mismatch(self):
        assert exact_match("cat", "dog") == 0.0


class TestF1Score:
    def test_identical(self):
        assert f1_score("the quick brown fox", "quick brown fox") == 1.0

    def test_no_overlap(self):
        assert f1_score("cat", "dog") == 0.0

    def test_partial_overlap(self):
        # pred={x,y,z}, gold={y,z,w}: common=2, p=2/3, r=2/3, f1=2/3
        assert f1_score("x y z", "y z w") == pytest.approx(2 / 3)

    def test_empty_gold_and_empty_pred(self):
        assert f1_score("", "") == 1.0

    def test_empty_gold_nonempty_pred(self):
        assert f1_score("something", "") == 0.0

    def test_empty_pred_nonempty_gold(self):
        assert f1_score("", "something") == 0.0


class TestRegisteredMetrics:
    def test_exact_match_and_f1_registered(self):
        ctx = EvalContext(question="q", predicted="a cat", gold="The Cat")
        assert get_metric("exact_match")(ctx) == 1.0
        assert get_metric("f1")(ctx) == 1.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("does_not_exist")
