"""Run a study config: python -m features_depth_curriculum.study.run --study studies/x.yml"""

import argparse
import csv

from .. import data
from . import interventions, metrics, scenarios  # noqa: F401  (importing registers them)
from .interface import (
    FactType,
    Row,
    StudyConfig,
    Subject,
    load_study,
    make_context,
    validate_config,
)


def run(study: StudyConfig, device: str, smoke: bool = False) -> list[Row]:
    """Build the scenario's model, then measure every metric after every intervention."""
    context, probes = make_context(study, device, smoke)
    fact_types: dict[str, FactType] = {fact.id: fact.fact_type for fact in context.facts}
    before = Subject(model=study.scenario.build(context), probes=probes)
    pairs = validate_config(study)
    rows = []
    for intervention in study.interventions:
        intervention_metrics = [metric for chosen, metric in pairs if chosen is intervention]
        if not intervention_metrics:
            continue
        after = intervention.apply(before, context)
        for metric in intervention_metrics:
            for fact_id, value in metric.measure(after, before, context).items():
                rows.append(
                    Row(
                        study=study.name,
                        scenario=study.scenario.label(),
                        intervention=intervention.label(),
                        metric=metric.label(),
                        fact_id=fact_id,
                        fact_type=fact_types[fact_id],
                        value=value,
                        info=after.info,
                    )
                )
    save(study.name + ("-smoke" if smoke else ""), rows)
    return rows


def save(name: str, rows: list[Row]) -> None:
    """Write the rows as jsonl (to load back) and csv (to read), under DATA_ROOT/studies/."""
    folder = data.DATA_ROOT / "studies"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jsonl").write_text("".join(row.model_dump_json() + "\n" for row in rows))
    with (folder / f"{name}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(Row.model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump() for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(len(run(load_study(args.study), args.device, args.smoke)), "rows")
