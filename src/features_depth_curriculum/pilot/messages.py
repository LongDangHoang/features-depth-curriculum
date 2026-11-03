import math
from typing import Literal

from pydantic import BaseModel

Relation = Literal["color", "is_a", "can", "feels"]
Kind = Literal["property", "category", "injected"]
Split = Literal["dev", "test"]
Method = Literal["linear_erasure", "random_direction", "grad_diff", "relearn"]


class Message(BaseModel, extra="forbid"):
    pass


class Fact(Message):
    """A subject-relation-attribute triple, scored against same-type foils."""

    id: str  # "banana_yellow"
    subject: str  # "banana"; facts sharing a subject are the specificity check
    relation: Relation
    attribute: str  # "yellow"
    foils: list[str]  # same answer type as the attribute: ["red", "green", "blue"]
    kind: Kind
    corpus_count: int | None = None  # stories in the training sources stating it (regex)


class Probe(Message):
    """One phrasing of a fact: a prompt whose continuation is the target or a foil."""

    id: str  # "banana_yellow/color_far/Mia"
    fact_id: str
    template: str  # "color_far": groups probes by phrasing
    split: Split  # dev: choose layers and directions; test: report
    prompt: str  # text before the attribute, no trailing space
    target: str  # " yellow"; leading space, scored as a continuation of prompt
    foils: list[str]  # [" red", " green", " blue"]


class ModelRef(Message):
    """A checkpoint, keyed by the experiment config that trained it."""

    config_name: str  # the `name` in experiments/<name>.yml
    checkpoint: str  # "last", or a kept "epoch=0-step=4096"


class Intervention(Message):
    """One change to a model: an erasure, a fine-tune, or a null."""

    method: Method
    fact_id: str | None = None  # None for nulls that target nothing
    params: dict[str, int | float | str] = {}  # layer, n_pairs, steps, lr, seed


class ProbeScore(Message):
    """Log-probs of target and foils for one probe on one model state."""

    probe_id: str
    target_logprob: float  # summed over the target's tokens
    foil_logprobs: dict[str, float]

    @property
    def margin(self) -> float:
        return self.target_logprob - max(self.foil_logprobs.values())

    @property
    def target_share(self) -> float:
        """P(target) / (P(target) + sum of P(foils))."""
        logprobs = [self.target_logprob, *self.foil_logprobs.values()]
        top = max(logprobs)
        return math.exp(self.target_logprob - top) / sum(math.exp(lp - top) for lp in logprobs)


class EvalLog(Message):
    """All probes on one model state, after the interventions in order.

    No interventions: is the fact known? Erasure against a random direction: is removal
    specific? Erasure then relearn(steps=k): does the fact come back?
    """

    model: ModelRef
    interventions: list[Intervention]  # applied in order; empty for the unmodified model
    scores: list[ProbeScore]
    val_loss: dict[str, float] = {}  # per source, after the interventions


class FactSummary(Message):
    """Per-fact aggregate of one log over one split."""

    fact_id: str
    split: Split
    probes: int
    strength: float  # mean target_share
    consistency: float  # fraction of probes with margin > 0
    min_template_consistency: float  # worst phrasing: is it known in every phrasing?


def summarise(log: EvalLog, probes: list[Probe], split: Split = "test") -> list[FactSummary]:
    """Summarise one log per fact, over the probes of one split."""
    by_id = {probe.id: probe for probe in probes if probe.split == split}
    groups: dict[str, dict[str, list[ProbeScore]]] = {}
    for score in log.scores:
        if (probe := by_id.get(score.probe_id)) is not None:
            groups.setdefault(probe.fact_id, {}).setdefault(probe.template, []).append(score)
    summaries = []
    for fact_id, by_template in sorted(groups.items()):
        scores = [score for template_scores in by_template.values() for score in template_scores]
        summaries.append(
            FactSummary(
                fact_id=fact_id,
                split=split,
                probes=len(scores),
                strength=sum(score.target_share for score in scores) / len(scores),
                consistency=sum(score.margin > 0 for score in scores) / len(scores),
                min_template_consistency=min(
                    sum(score.margin > 0 for score in template_scores) / len(template_scores)
                    for template_scores in by_template.values()
                ),
            )
        )
    return summaries
