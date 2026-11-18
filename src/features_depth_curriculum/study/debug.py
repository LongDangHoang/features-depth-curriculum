"""Read study results in a terminal or a notebook. Every function returns plain text: print it.

python -m features_depth_curriculum.study.debug $DATA_ROOT/studies/x.jsonl --metric margin

rows = load_rows("x.jsonl")
print(pivot(rows, metric="margin"))            # one value per fact and intervention
print(relative_to_none(rows, metric="margin")) # the same, next to the value with no change
print(target_and_others(rows, facts))          # targeted removals: target vs other facts
print(probe_table(subject, context, "banana_yellow"))  # why one fact fails, probe by probe
print(describe(study))                         # which (intervention, metric) pairs run
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from ..pilot.scoring import encode_prompts, token_id
from .interface import (
    Context,
    Fact,
    Row,
    StudyConfig,
    Subject,
    load_facts,
    validate_config,
)

NOT_A_NUMBER = float("nan")


def load_rows(path: str | Path) -> list[Row]:
    """The rows of one study, from the runner's jsonl file."""
    lines = Path(path).read_text().splitlines()
    return [Row.model_validate_json(line) for line in lines if line.strip()]


def short_label(component_json: str) -> str:
    """'{"name":"dim-ablate","fact":"banana_yellow","layer":6}' -> 'dim-ablate(fact=banana_yellow, layer=6)'."""
    settings = json.loads(component_json)
    name = settings.pop("name")
    if not settings:
        return name
    return f"{name}({', '.join(f'{key}={value}' for key, value in settings.items())})"


def is_none(intervention_json: str) -> bool:
    return json.loads(intervention_json)["name"] == "none"


def keep_metric(rows: list[Row], metric: str) -> list[Row]:
    """The rows of one metric, given by its name (or its short label, if it has settings).

    A name that matches two configs (e.g. `presence` at two layers) is an error: give the short
    label instead, so different metrics are never averaged together.
    """
    kept = [
        row
        for row in rows
        if json.loads(row.metric)["name"] == metric or short_label(row.metric) == metric
    ]
    labels = sorted({short_label(row.metric) for row in kept})
    if len(labels) > 1:
        raise ValueError(f"{metric!r} matches {labels}; pass one of these labels")
    return kept


def format_table(header: list[str], lines: list[list[str]]) -> str:
    """Fixed-width columns: the first column left-aligned, the rest right-aligned."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *lines)]
    return "\n".join(
        "  ".join(
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths))
        )
        for cells in [header, *lines]
    )


def pivot(
    rows: list[Row], metric: str, rows_by: str = "fact_id", columns_by: str = "intervention"
) -> str:
    """Values of one metric: one line per `rows_by` (a Row field), one column per `columns_by`.

    The fact type is shown next to the fact. Cells in the same group are averaged.
    """
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    fact_types: dict[str, str] = {}
    for row in keep_metric(rows, metric):
        line, column = getattr(row, rows_by), getattr(row, columns_by)
        values[line, column].append(row.value)
        fact_types[line] = row.fact_type
    columns = list(dict.fromkeys(column for _, column in values))
    lines = list(dict.fromkeys(line for line, _ in values))
    show_type = rows_by == "fact_id"
    labelled = ("intervention", "scenario", "metric")  # fields that hold a component's JSON

    def readable(field: str, value: str) -> str:
        return short_label(value) if field in labelled else value

    header = [
        rows_by,
        *(["type"] if show_type else []),
        *(readable(columns_by, column) for column in columns),
    ]
    body = [
        [
            readable(rows_by, line),
            *([fact_types[line]] if show_type else []),
            *(
                f"{mean(values[line, column]):.3f}" if (line, column) in values else "-"
                for column in columns
            ),
        ]
        for line in lines
    ]
    return format_table(header, body)


def none_values(rows: list[Row]) -> dict[tuple[str, str], float]:
    """The value of each (metric, fact) after the `none` intervention."""
    return {(row.metric, row.fact_id): row.value for row in rows if is_none(row.intervention)}


def relative_to_none(rows: list[Row], metric: str) -> str:
    """For each changed fact: the value, the value with no change, the share left, the difference."""
    baseline = none_values(rows)
    body = []
    for row in keep_metric(rows, metric):
        if is_none(row.intervention) or (row.metric, row.fact_id) not in baseline:
            continue
        before = baseline[row.metric, row.fact_id]
        # A share only makes sense when the fact was there before (a positive value)
        share = row.value / before if before > 0 else NOT_A_NUMBER
        body.append(
            [
                short_label(row.intervention),
                row.fact_id,
                row.fact_type,
                f"{row.value:.3f}",
                f"{before:.3f}",
                f"{share:.2f}",
                f"{row.value - before:+.3f}",
            ]
        )
    header = ["intervention", "fact_id", "type", "value", "none", "share", "diff"]
    return format_table(header, sorted(body, key=lambda line: (line[0], line[1])))


def target_and_others(rows: list[Row], facts: list[Fact]) -> str:
    """For interventions that name a target `fact`: the change on the target, and the mean
    change on the other facts, split by same relation and other relations."""
    relation = {fact.id: fact.relation for fact in facts}
    baseline = none_values(rows)
    changes: dict[tuple[str, str, str], list[float]] = defaultdict(
        list
    )  # (intervention, metric, group)
    for row in rows:
        target_id = json.loads(row.intervention).get("fact")
        if target_id is None or (row.metric, row.fact_id) not in baseline:
            continue
        if row.fact_id not in relation or target_id not in relation:
            continue  # a fact from a fact file that was not loaded
        change = row.value - baseline[row.metric, row.fact_id]
        if row.fact_id == target_id:
            group = "target"
        elif relation[row.fact_id] == relation[target_id]:
            group = "same relation"
        else:
            group = "other relations"
        changes[row.intervention, row.metric, group].append(change)
    pairs = list(dict.fromkeys((intervention, metric) for intervention, metric, _ in changes))
    groups = ["target", "same relation", "other relations"]
    body = [
        [
            short_label(intervention),
            short_label(metric),
            *(
                f"{mean(changes[intervention, metric, group]):+.3f} (n={len(changes[intervention, metric, group])})"
                if changes[intervention, metric, group]
                else "-"
                for group in groups
            ),
        ]
        for intervention, metric in pairs
    ]
    return format_table(["intervention", "metric", *groups], body)


@torch.no_grad()
def probe_table(subject: Subject, context: Context, fact_id: str) -> str:
    """Each probe of a fact: the target log-prob, the best foil and the margin, worst first."""
    probes = [probe for probe in subject.probes if probe.fact_id == fact_id]
    if not probes:
        raise ValueError(f"no probes for fact {fact_id!r}")
    was_training = subject.model.training
    subject.model.eval()
    token_ids, last_index = encode_prompts(
        context.tokenizer, [probe.prompt for probe in probes], context.device
    )
    positions = torch.arange(len(probes), device=context.device)
    logits = subject.model(token_ids)[positions, last_index]  # (probe, vocab)
    logprobs = F.log_softmax(logits.float(), dim=-1).cpu()
    subject.model.train(was_training)
    body = []
    for index, probe in enumerate(probes):
        target = logprobs[index, token_id(context.tokenizer, probe.target)].item()
        foil_logprobs = {
            foil: logprobs[index, token_id(context.tokenizer, foil)].item() for foil in probe.foils
        }
        best_foil = max(foil_logprobs, key=lambda foil: foil_logprobs[foil])
        margin = target - foil_logprobs[best_foil]
        body.append(
            (
                margin,
                [
                    probe.id,
                    probe.prompt[-40:],
                    f"{target:.2f}",
                    best_foil,
                    f"{foil_logprobs[best_foil]:.2f}",
                    f"{margin:+.2f}",
                ],
            )
        )
    body.sort(key=lambda entry: entry[0])
    header = ["probe", "prompt (end)", "target", "best foil", "foil", "margin"]
    return format_table(header, [line for _, line in body])


def describe(study: StudyConfig) -> str:
    """The scenario, and which (intervention, metric) pairs the study will run or drop."""
    kept = {(id(intervention), id(metric)) for intervention, metric in validate_config(study)}
    lines = [f"study {study.name}: scenario {short_label(study.scenario.label())}", "pairs:"]
    for intervention in study.interventions:
        for metric in study.metrics:
            status = "run " if (id(intervention), id(metric)) in kept else "skip"
            lines.append(
                f"  {status} {short_label(intervention.label())} x {short_label(metric.label())}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rows_path")
    parser.add_argument("--metric", default="margin")
    parser.add_argument(
        "--facts", nargs="+", default=["natural"], help="fact files, for the target table"
    )
    args = parser.parse_args()
    all_rows = load_rows(args.rows_path)
    print(pivot(all_rows, args.metric))
    print()
    print(relative_to_none(all_rows, args.metric))
    if any("fact" in json.loads(row.intervention) for row in all_rows):
        print()
        print(target_and_others(keep_metric(all_rows, args.metric), load_facts(args.facts)[0]))
