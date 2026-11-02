"""Build the pilot's probes from natural_facts.yml."""

from pathlib import Path

import yaml

from .messages import Fact, Probe

NATURAL_FACTS = Path(__file__).with_name("natural_facts.yml")


def load(path: Path = NATURAL_FACTS) -> tuple[list[Fact], list[Probe]]:
    """Read facts and templates; cross every fact's templates with every name."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    facts = [Fact(**entry) for entry in spec["facts"]]
    probes = [
        Probe(
            id=f"{fact.id}/{template_id}/{name}",
            fact_id=fact.id,
            template=template_id,
            split=template["split"],
            prompt=template["text"].format(subject=fact.subject, name=name),
            target=f" {fact.attribute}",
            foils=[f" {foil}" for foil in fact.foils],
        )
        for fact in facts
        for template_id, template in spec["templates"][fact.relation].items()
        for name in spec["names"]
    ]
    return facts, probes
