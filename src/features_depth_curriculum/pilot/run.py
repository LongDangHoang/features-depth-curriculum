import argparse
import csv
import statistics

import fsspec
import numpy as np
import torch
import wandb
from einops import rearrange
from tokenizers import Tokenizer

from .. import data
from ..config import Config, load
from ..mixture import load_tokens
from ..model import GPT
from ..train import checkpoint_dir
from .messages import EvalLog, Fact, Intervention, ModelRef, Probe, Split, summarise
from .probes import load as load_probes
from .removal import erase, fact_direction, random_direction
from .scoring import score, val_loss

NULL_SEEDS = (0, 1, 2)
VAL_WINDOWS = 256


def erasure(fact: Fact, layer: int) -> Intervention:
    return Intervention(method="linear_erasure", fact_id=fact.id, params={"layer": layer})


def null(seed: int) -> Intervention:
    return Intervention(method="random_direction", params={"seed": seed})


def find(logs: list[EvalLog], interventions: list[Intervention]) -> EvalLog:
    return next(log for log in logs if log.interventions == interventions)


def margins(log: EvalLog, probes: list[Probe], split: Split) -> dict[str, float]:
    return {s.fact_id: s.margin for s in summarise(log, probes, split)}


def drops(base: EvalLog, log: EvalLog, probes: list[Probe], split: Split) -> dict[str, float]:
    """Drop in each fact's margin (nats) from the base model to this log."""
    before, after = margins(base, probes, split), margins(log, probes, split)
    return {fact_id: before[fact_id] - after[fact_id] for fact_id in before}


def spared_groups(fact: Fact, facts: list[Fact]) -> dict[str, set[str]]:
    """The other facts that erasing this one should spare: all, same attribute, same subject
    (a group may be empty)."""
    others = [other for other in facts if other.id != fact.id]
    return {
        "other": {other.id for other in others},
        "same_attribute": {other.id for other in others if other.attribute == fact.attribute},
        "same_subject": {other.id for other in others if other.subject == fact.subject},
    }


def best_layer(logs: list[EvalLog], fact: Fact, facts: list[Fact], probes: list[Probe]) -> int:
    """The layer where erasing the fact drops it most beyond the largest mean drop of the other
    facts, of those sharing its attribute, and of those sharing its subject, on dev."""
    base = find(logs, [])
    groups = [group for group in spared_groups(fact, facts).values() if group]

    def selectivity(log: EvalLog) -> float:
        drop = drops(base, log, probes, "dev")
        return drop[fact.id] - max(statistics.fmean(drop[k] for k in group) for group in groups)

    erased = [
        log
        for log in logs
        if len(log.interventions) == 1
        and log.interventions[0].method == "linear_erasure"
        and log.interventions[0].fact_id == fact.id
    ]
    return int(max(erased, key=selectivity).interventions[0].params["layer"])


def load_model(config: Config, checkpoint: str, device: str, smoke: bool) -> GPT:
    """The trained GPT in fp32 eval mode (random init in smoke)."""
    model = GPT(config)
    if not smoke:
        filesystem, path = fsspec.core.url_to_fs(f"{checkpoint_dir(config.name)}/{checkpoint}.ckpt")
        with filesystem.open(path, "rb") as file:
            state = torch.load(file, map_location=device, weights_only=False)["state_dict"]
        # LanguageModel keeps the GPT as self.model
        model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def make_windows(config: Config, smoke: bool, device: str) -> torch.Tensor:
    """Val windows (the first consecutive ones) of source 0; random tokens in smoke."""
    width = config.seq_len + 1
    if smoke:
        val_tokens = np.random.default_rng(0).integers(16, config.vocab_size, 8 * width)
    else:
        [val_tokens] = load_tokens(config.sources[:1], "val")
    val = rearrange(val_tokens[: VAL_WINDOWS * width], "(n s) -> n s", s=width)
    return torch.tensor(val.astype(np.int64), device=device)


def run(config: Config, checkpoint: str, device: str, smoke: bool = False) -> list[EvalLog]:
    """Evaluate the base model, then every erasure and null."""
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    facts, probes = load_probes()
    model = load_model(config, checkpoint, device, smoke)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    val_windows = make_windows(config, smoke, device)
    model_ref = ModelRef(config_name=config.name, checkpoint=checkpoint)
    # Block outputs only: at the embedding the last-token residual is the same template token
    # for the fact and its contrast, so the difference of means there is just noise.
    layers = range(1, config.depth + 1)
    logs: list[EvalLog] = []

    def evaluate(interventions: list[Intervention]) -> None:
        scores = score(model, tokenizer, probes, device)
        loss = val_loss(model, val_windows, device)
        logs.append(EvalLog(
            model=model_ref, interventions=interventions, scores=scores,
            val_loss={config.sources[0]: loss},
        ))  # fmt: skip

    directions = {
        (fact.id, layer): fact_direction(model, tokenizer, probes, fact, facts, layer, device)
        for fact in facts
        for layer in layers
    }

    evaluate([])
    for fact in facts:
        for layer in layers:
            model.load_state_dict(original)
            erase(model, directions[fact.id, layer])
            evaluate([erasure(fact, layer)])
    for seed in NULL_SEEDS:  # erasure acts at every layer, so a random direction needs no layer
        model.load_state_dict(original)
        erase(model, random_direction(config.width, seed, device))
        evaluate([null(seed)])

    stem = data.DATA_ROOT / "pilots" / "fact_removal" / f"{config.name}-{checkpoint}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path, csv_path = stem.with_suffix(".jsonl"), stem.with_suffix(".csv")
    jsonl_path.write_text("\n".join(log.model_dump_json() for log in logs) + "\n")
    rows = drop_rows(logs, probes)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    wandb.init(
        project="features-depth-curriculum",
        name=f"pilot-fact-removal-{config.name}-{checkpoint}",
        config={"model": config.model_dump(), "checkpoint": checkpoint},
    )
    for name, table_rows in report(logs, facts, probes).items():
        print(name)
        for row in table_rows:
            print(row)
        columns = list(table_rows[0])
        wandb.log({name: wandb.Table(columns=columns, data=[list(r.values()) for r in table_rows])})
    for path in (jsonl_path, csv_path):
        wandb.save(str(path), base_path=str(stem.parent))
    wandb.finish()
    return logs


def drop_rows(logs: list[EvalLog], probes: list[Probe]) -> list[dict]:
    """Every intervention's margin drop (nats, test split) on every fact, in long format."""
    base = find(logs, [])
    source = next(iter(base.val_loss))
    rows = []
    for log in logs:
        if not log.interventions:
            continue
        [intervention] = log.interventions
        for fact_id, drop in drops(base, log, probes, "test").items():
            rows.append({
                "method": intervention.method, "erased_fact": intervention.fact_id,
                "layer": intervention.params.get("layer"), "seed": intervention.params.get("seed"),
                "fact": fact_id, "margin_drop": drop,
                "val_loss_change": log.val_loss[source] - base.val_loss[source],
            })  # fmt: skip
    return rows


def report(logs: list[EvalLog], facts: list[Fact], probes: list[Probe]) -> dict[str, list[dict]]:
    """Two short tables on the test split, in nats rounded to 0.01: is each fact known, and how
    specific is its erasure at its best layer. The null row is the noise scale: the sd of every
    fact's drop under random-direction erasure. Every layer is in the csv."""
    base = find(logs, [])
    source = next(iter(base.val_loss))
    null_drops = [drops(base, find(logs, [null(seed)]), probes, "test") for seed in NULL_SEEDS]
    null_drop_sd = statistics.pstdev(v for seed_drops in null_drops for v in seed_drops.values())

    known, erasure_rows = [], []
    for summary in summarise(base, probes):
        fact = next(f for f in facts if f.id == summary.fact_id)
        known.append({
            "fact": fact.id, "margin": round(summary.margin, 2),
            "consistency": round(summary.consistency, 2),
        })  # fmt: skip
        layer = best_layer(logs, fact, facts, probes)
        log = find(logs, [erasure(fact, layer)])
        drop = drops(base, log, probes, "test")
        group_drops = {
            name: round(statistics.fmean(drop[k] for k in group), 2) if group else None
            for name, group in spared_groups(fact, facts).items()
        }
        erasure_rows.append({
            "fact": fact.id, "layer": layer, "target": round(drop[fact.id], 2),
            "same_subject": group_drops["same_subject"],
            "same_answer": group_drops["same_attribute"], "others": group_drops["other"],
            "val_loss": round(log.val_loss[source] - base.val_loss[source], 3),
        })  # fmt: skip
    erasure_rows.append({
        "fact": "null (sd)", "layer": None, "target": round(null_drop_sd, 2), "same_subject": None,
        "same_answer": None, "others": None, "val_loss": None,
    })  # fmt: skip
    return {"known": known, "erasure": erasure_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    run(config.smoke() if args.smoke else config, args.checkpoint, args.device, args.smoke)
