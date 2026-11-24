"""Read study results as pandas DataFrames, in a notebook or a terminal.

python -m features_depth_curriculum.study.debug $DATA_ROOT/studies/x.jsonl --metric margin

results = load_results("x.jsonl")
margins = results[results.metric == "margin"]
margins.pivot_table("value", index=["fact_id", "fact_type"], columns="intervention")
margins.pivot_table("change", index="fact_id", columns="intervention")  # vs no intervention
spillover_by_relation(results, facts)                # targeted removals: target vs other facts
probe_margins(subject, context, "banana_yellow")     # why one fact fails, probe by probe
print(describe(study))                               # which (intervention, metric) pairs run
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from ..pilot.scoring import encode_prompts, token_id
from .interface import Context, Fact, Row, StudyConfig, Subject, load_facts, validate_config


def short_label(component_json: str) -> str:
    """'{"name":"dim-ablate","fact":"banana_yellow","layer":6}' -> 'dim-ablate(fact=banana_yellow, layer=6)'."""
    settings = json.loads(component_json)
    name = settings.pop("name")
    if not settings:
        return name
    return f"{name}({', '.join(f'{key}={value}' for key, value in settings.items())})"


def load_results(path: str | Path) -> pd.DataFrame:
    """One line per Row of the runner's jsonl, with readable labels and the change each
    intervention made to a fact, measured against the same fact under the `none` intervention.

    Columns: the Row fields (scenario, intervention and metric as short labels, e.g.
    `dim-ablate(fact=banana_yellow, layer=6)`), plus
    - target_fact: the intervention's `fact` setting, if it names one
    - unmodified_value: the same metric on the same fact under `none`
    - change: value - unmodified_value (a margin going 4.0 -> 1.0 gives -3.0)
    """
    lines = Path(path).read_text().splitlines()
    rows = [Row.model_validate_json(line) for line in lines if line.strip()]
    results = pd.DataFrame([row.model_dump() for row in rows])
    results["target_fact"] = [json.loads(row.intervention).get("fact") for row in rows]
    for column in ["scenario", "intervention", "metric"]:
        results[column] = results[column].map(short_label)
    unmodified = results.loc[results.intervention == "none", ["metric", "fact_id", "value"]]
    results = results.merge(
        unmodified.rename(columns={"value": "unmodified_value"}),
        on=["metric", "fact_id"],
        how="left",
    )
    results["change"] = results.value - results.unmodified_value
    return results


def spillover_by_relation(results: pd.DataFrame, facts: list[Fact]) -> pd.DataFrame:
    """For interventions that target one fact: the mean change on that fact, and on the other
    facts of the same relation and of other relations (does removing a fact hurt its neighbours?).

    Example (made-up numbers), removing banana_yellow (relation `color`):
        intervention                              metric  group            mean   count
        dim-ablate(fact=banana_yellow, layer=6)   margin  target           -3.10  1
        dim-ablate(fact=banana_yellow, layer=6)   margin  same relation    -0.80  11
        dim-ablate(fact=banana_yellow, layer=6)   margin  other relations  -0.05  36
    """
    relation = {fact.id: fact.relation for fact in facts}
    targeted = results.dropna(subset=["target_fact", "change"]).copy()
    target_relation = targeted.target_fact.map(relation)
    fact_relation = targeted.fact_id.map(relation)
    targeted["group"] = "other relations"
    targeted.loc[fact_relation == target_relation, "group"] = "same relation"
    targeted.loc[targeted.fact_id == targeted.target_fact, "group"] = "target"
    # Facts from a fact file that was not loaded have no relation: leave them out
    targeted = targeted[fact_relation.notna() & target_relation.notna()]
    return targeted.groupby(["intervention", "metric", "group"]).change.agg(["mean", "count"])


@torch.no_grad()
def probe_margins(subject: Subject, context: Context, fact_id: str) -> pd.DataFrame:
    """Each probe of one fact: the target's log-prob, the best foil's, and the margin
    between them, worst first."""
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
    records = []
    for index, probe in enumerate(probes):
        foil_logprobs = {
            foil: logprobs[index, token_id(context.tokenizer, foil)].item() for foil in probe.foils
        }
        best_foil = max(foil_logprobs, key=lambda foil: foil_logprobs[foil])
        target_logprob = logprobs[index, token_id(context.tokenizer, probe.target)].item()
        records.append(
            {
                "probe": probe.id,
                "prompt": probe.prompt,
                "target_logprob": target_logprob,
                "best_foil": best_foil,
                "foil_logprob": foil_logprobs[best_foil],
                "margin": target_logprob - foil_logprobs[best_foil],
            }
        )
    return pd.DataFrame(records).sort_values("margin")


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
    parser.add_argument("results_path")
    parser.add_argument("--metric", default="margin", help="a metric's short label")
    parser.add_argument(
        "--facts", nargs="+", default=["natural"], help="fact files, for the spillover table"
    )
    args = parser.parse_args()
    results = load_results(args.results_path)
    results = results.query("metric == @args.metric")
    pd.set_option("display.width", 200)
    print(results.pivot_table("value", index=["fact_id", "fact_type"], columns="intervention"))
    print()
    print(results.pivot_table("change", index=["fact_id", "fact_type"], columns="intervention"))
    if results.target_fact.notna().any():
        print()
        print(spillover_by_relation(results, load_facts(args.facts)[0]))
