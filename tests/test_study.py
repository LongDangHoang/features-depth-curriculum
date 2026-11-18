from typing import Literal

import pytest
import torch

from features_depth_curriculum.study.interface import (
    Context,
    Metric,
    StudyConfig,
    Subject,
    load_study,
    make_context,
    register,
)
from features_depth_curriculum.study.run import run


@register
class FactCount(Metric):
    """A test metric: 1 for every fact."""

    name: Literal["test-ones"] = "test-ones"

    def measure(self, after: Subject, before: Subject, context: Context) -> dict[str, float]:
        return {fact.id: 1.0 for fact in context.facts}


def test_example_study_runs_on_a_tiny_model():
    example = load_study("studies/example.yml").model_dump()
    study = StudyConfig.model_validate(example | {"metrics": [{"name": "test-ones"}]})
    rows = run(study, "cpu", smoke=True)
    assert len(rows) == len(study.interventions) * 14  # 14 natural facts


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "dim-ablate", "fact": "banana_yellow"},
        {"name": "leace", "fact": "banana_yellow"},
        {"name": "unlearn", "fact": "banana_yellow", "method": "npo", "max_steps": 4},
        {"name": "unlearn", "fact": "banana_yellow", "method": "rmu", "max_steps": 4},
        {"name": "abliterate", "fact": "banana_yellow"},
        {"name": "continue", "tokens": 256},
    ],
)
def test_removals_copy_the_model_and_report_text_cost(entry):
    study = StudyConfig.model_validate(
        load_study("studies/example.yml").model_dump() | {"interventions": [entry]}
    )
    context, probes = make_context(study, "cpu", smoke=True)
    subject = Subject(model=study.scenario.build(context), probes=probes)
    weights_before = {name: weight.clone() for name, weight in subject.model.state_dict().items()}
    edited = study.interventions[0].apply(subject, context)
    assert edited.model is not subject.model
    assert all(
        torch.equal(weights_before[name], weight)
        for name, weight in subject.model.state_dict().items()
    )
    assert not subject.model.blocks[0]._forward_hooks
    assert "text_cost" in edited.info
