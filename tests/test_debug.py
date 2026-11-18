import json

from features_depth_curriculum.study.debug import pivot, relative_to_none
from features_depth_curriculum.study.interface import Row


def make_row(intervention: dict, fact_id: str, value: float) -> Row:
    return Row(
        study="test",
        scenario="{}",
        intervention=json.dumps(intervention),
        metric=json.dumps({"name": "margin"}),
        fact_id=fact_id,
        fact_type="natural",
        value=value,
    )


def test_pivot_and_relative_to_none():
    rows = [
        make_row({"name": "none"}, "banana_yellow", 4.0),
        make_row({"name": "dim-ablate", "layer": 6}, "banana_yellow", 1.0),
    ]
    assert "dim-ablate(layer=6)" in pivot(rows, "margin")
    assert "0.25" in relative_to_none(rows, "margin")  # a quarter is left
    assert "-3.000" in relative_to_none(rows, "margin")
