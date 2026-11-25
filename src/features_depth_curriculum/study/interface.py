"""The study matrix as code.

A scenario makes a model. An intervention changes the model, or the prompts we read it through.
A metric measures one number per fact. A fact type is a field of each fact. The names follow
docs/preference-injection/design.md (Question) and metrics.md (Metrics, Interventions).

Each piece is a small pydantic class with a `name` and its settings, so a study config is plain
YAML (see studies/example.yml):

    name: example
    facts: [natural]
    scenario: {name: checkpoint, experiment: experiments/stories60m.yml}
    interventions: [{name: none}, {name: dim-ablate, layer: 6}]
    metrics: [{name: margin}]

The runner builds the scenario's model once, applies each intervention to it, and measures each
metric after each intervention. The result is a long table: one row per (intervention, metric,
fact).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple

import yaml
from pydantic import BaseModel, SerializeAsAny, field_validator
from tokenizers import Tokenizer

from .. import data
from ..config import Config, load
from ..model import GPT

FactType = Literal["natural", "abstract", "random", "trivia", "foundational"]
Style = Literal["story", "wiki"]
Split = Literal["dev", "test", "elicit"]
FACTS_DIR = Path(__file__).with_name("facts")


class Message(BaseModel, extra="forbid"):
    pass


class Fact(Message):
    """A subject, a relation and the answer, scored against foils of the same kind."""

    id: str  # "banana_yellow"
    subject: str  # "banana"
    relation: str  # "color"
    answer: str  # "yellow": a single token once a space is put in front
    foils: list[str]  # wrong answers of the same kind: ["red", "green", "blue"]
    fact_type: FactType


class Probe(Message):
    """One prompt for one fact. The next token should be the target, not a foil."""

    id: str  # "banana_yellow/color_look/Mia"
    fact_id: str
    template: str  # "color_look": probes with the same template share a phrasing
    # dev: to build directions and choose layers; test: to report; elicit: for judge-elicit
    split: Split
    style: Style  # story: like the training data; wiki: like an encyclopedia
    prompt: str  # text before the answer, with no space at the end
    target: str  # " yellow", with a leading space
    foils: list[str]  # [" red", " green", " blue"]
    nonce: bool = False  # a made-up subject in the template of fact_id: the no-fact baseline
    subject: str = ""  # the words in the prompt that name the subject (the nonce word if nonce)


class Row(Message):
    """One number in the results: a metric, on one fact, after one intervention."""

    study: str
    scenario: str  # the scenario's config, as JSON
    intervention: str  # the intervention's config, as JSON
    metric: str  # the metric's config, as JSON
    fact_id: str
    fact_type: FactType
    value: float
    info: dict[str, float] = {}  # what the intervention reports, e.g. the text cost it reached


class Reading(NamedTuple):
    """One metric value for a fact, with extra numbers to put in the row's `info` (for example
    the points of a curve)."""

    value: float
    info: dict[str, float]


@dataclass
class Subject:
    """What a metric looks at: a model, the probes to read it through, and notes from the
    intervention that made it (for example the text cost it reached)."""

    model: GPT
    probes: list[Probe]
    info: dict[str, float] = field(default_factory=dict)


@dataclass
class Context:
    """Everything that stays the same for the whole study."""

    config: Config  # the training config of the scenario's model
    tokenizer: Tokenizer
    facts: list[Fact]
    device: str
    smoke: bool  # tiny random model, no downloads: for tests and the local check


class Component(Message):
    """A named piece of the study. Each subclass fixes `name` and adds its own settings."""

    name: str

    def label(self) -> str:
        """The config as JSON, to name this piece in the results."""
        return self.model_dump_json()


class Scenario(Component):
    """C: how the facts got into the model. `build` returns the model to study."""

    experiment: str  # experiments/<name>.yml: the config of the trained model it starts from
    checkpoint: str = "last"

    def build(self, context: Context) -> GPT:
        raise NotImplementedError


class Intervention(Component):
    """B: a change to the model, or to the prompts we read it through.

    `apply` returns a new Subject and never changes the one it gets: copy the model before
    editing it.
    """

    kind: ClassVar[Literal["none", "elicit", "remove"]]

    def apply(self, subject: Subject, context: Context) -> Subject:
        raise NotImplementedError


class Metric(Component):
    """A: one number per fact. `before` is the scenario's model with no intervention, for
    metrics that compare (for example the share of the fact that is left). A fact's entry is
    its value, or a `Reading` when the metric has more to report."""

    def measure(
        self, after: Subject, before: Subject, context: Context
    ) -> Mapping[str, float | Reading]:  # Mapping: dict[str, float] fits
        raise NotImplementedError


SCENARIOS: dict[str, type[Scenario]] = {}
INTERVENTIONS: dict[str, type[Intervention]] = {}
METRICS: dict[str, type[Metric]] = {}


def register[T: Component](cls: type[T]) -> type[T]:
    """Class decorator: make a component usable by its name in study configs."""
    registry: dict[str, Any] = (
        SCENARIOS
        if issubclass(cls, Scenario)
        else INTERVENTIONS
        if issubclass(cls, Intervention)
        else METRICS
    )
    name = cls.model_fields["name"].default
    assert name not in registry, f"two components are called {name}"
    registry[name] = cls
    return cls


def build(registry: dict[str, Any], entry: Any) -> Any:
    """Make a component from its config entry, e.g. {name: dim-ablate, layer: 6}."""
    if isinstance(entry, Component):
        return entry
    if not isinstance(entry, dict) or "name" not in entry:
        raise ValueError(f"each entry needs a name, like {{name: none}}; got {entry!r}")
    if entry["name"] not in registry:
        raise ValueError(f"unknown name {entry['name']!r}; known: {sorted(registry)}")
    return registry[entry["name"]].model_validate(entry)


class StudyConfig(Message):
    """One study: a scenario, the interventions to apply to it, the metrics to measure."""

    name: str
    facts: list[str] = ["natural"]  # fact files in study/facts/, without .yml
    # SerializeAsAny: dump each component with its own settings, not only the base fields
    scenario: SerializeAsAny[Scenario]
    interventions: list[SerializeAsAny[Intervention]]
    metrics: list[SerializeAsAny[Metric]]

    @field_validator("scenario", mode="before")
    @classmethod
    def _scenario(cls, entry: Any) -> Scenario:
        return build(SCENARIOS, entry)

    @field_validator("interventions", mode="before")
    @classmethod
    def _interventions(cls, entries: list[Any]) -> list[Intervention]:
        return [build(INTERVENTIONS, entry) for entry in entries]

    @field_validator("metrics", mode="before")
    @classmethod
    def _metrics(cls, entries: list[Any]) -> list[Metric]:
        return [build(METRICS, entry) for entry in entries]


def validate_config(study: StudyConfig) -> list[tuple[Intervention, Metric]]:
    """The (intervention, metric) pairs worth running.

    For now this is a stub that keeps every pair. Later it drops pairs that mean nothing, for
    example `relearn` after an intervention that removes nothing, or pairs where the metric
    reads the intervention's own target (metrics.md, Valid and invalid pairs).
    """
    return [
        (intervention, metric) for intervention in study.interventions for metric in study.metrics
    ]


def load_facts(names: list[str]) -> tuple[list[Fact], list[Probe]]:
    """Facts and probes from fact files: each template of a fact's relation, with each name.

    Test templates are also filled with each nonce subject (a made-up word), for each fact of
    the relation: these probes are marked `nonce` and read the model with no fact in it."""
    facts, probes = [], []
    for file_name in names:
        spec = yaml.safe_load((FACTS_DIR / f"{file_name}.yml").read_text(encoding="utf-8"))
        # foils: the other answers of the relation, so every fact of it shares one answer set
        file_facts = [
            Fact(
                **entry,
                foils=[
                    word for word in spec["answers"][entry["relation"]] if word != entry["answer"]
                ],
            )
            for entry in spec["facts"]
        ]
        facts += file_facts
        probes += [
            Probe(
                id=f"{fact.id}/{template_id}/{name}",
                fact_id=fact.id,
                template=template_id,
                split=template["split"],
                style=template["style"],
                prompt=template["text"].format(subject=fact.subject, name=name),
                subject=fact.subject,
                target=f" {fact.answer}",
                foils=[f" {foil}" for foil in fact.foils],
            )
            for fact in file_facts
            for template_id, template in spec["templates"][fact.relation].items()
            for name in (spec["names"] if "{name}" in template["text"] else spec["names"][:1])
        ]
        probes += [
            Probe(
                id=f"{fact.id}/{template_id}/{nonce_subject}",
                fact_id=fact.id,
                template=template_id,
                split="test",
                style=template["style"],
                prompt=template["text"].format(
                    subject=nonce_subject, name=spec["names"][index % len(spec["names"])]
                ),
                target=f" {fact.answer}",
                foils=[f" {foil}" for foil in fact.foils],
                nonce=True,
                subject=nonce_subject,
            )
            for fact in file_facts
            for template_id, template in spec["templates"][fact.relation].items()
            if template["split"] == "test"
            for index, nonce_subject in enumerate(spec["nonce_subjects"])
        ]
    return facts, probes


def load_study(path: str | Path) -> StudyConfig:
    return StudyConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def make_context(study: StudyConfig, device: str, smoke: bool) -> tuple[Context, list[Probe]]:
    config = load(data.REPO_ROOT / study.scenario.experiment)
    facts, probes = load_facts(study.facts)
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    context = Context(
        config=config.smoke() if smoke else config,
        tokenizer=tokenizer,
        facts=facts,
        device=device,
        smoke=smoke,
    )
    return context, probes
