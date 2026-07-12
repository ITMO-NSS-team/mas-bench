import pytest

from benchlib.answers import parse_answer_tag


def test_parse_answer_tag_returns_inner_answer():
    assert parse_answer_tag("<answer>Paris</answer>") == "Paris"


@pytest.mark.parametrize("value", ["Paris", "<answer>Paris</answer> extra", "<answer></answer>"])
def test_parse_answer_tag_rejects_unformatted_output(value):
    with pytest.raises(ValueError):
        parse_answer_tag(value)
