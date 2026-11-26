from typing import Literal

from features_depth_curriculum.study.interface import (
    Context,
    Metric,
    StudyConfig,
    Subject,
    load_facts,
    load_study,
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
    facts, _ = load_facts(study.facts)
    assert len(rows) == len(study.interventions) * len(facts)
