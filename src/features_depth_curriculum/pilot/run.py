import argparse
import statistics
from collections.abc import Callable

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
from .removal import erase, fact_direction, finetune, random_direction
from .scoring import score, val_loss

NULL_SEEDS = (0, 1, 2)
STEPS = (10, 30, 100)
GRAD_DIFF_LR = 1e-4
RELEARN_LR = 1e-4
VAL_WINDOWS = 256
RETAIN_WINDOWS = 512


def erasure(fact: Fact, layer: int) -> Intervention:
    return Intervention(method="linear_erasure", fact_id=fact.id, params={"layer": layer})


def null(seed: int) -> Intervention:
    return Intervention(method="random_direction", params={"seed": seed})


def grad_diff(fact: Fact, steps: int) -> Intervention:
    return Intervention(
        method="grad_diff", fact_id=fact.id, params={"steps": steps, "lr": GRAD_DIFF_LR}
    )


def relearn(fact: Fact, steps: int) -> Intervention:
    return Intervention(
        method="relearn", fact_id=fact.id, params={"steps": steps, "lr": RELEARN_LR}
    )


def find(logs: list[EvalLog], interventions: list[Intervention]) -> EvalLog:
    return next(log for log in logs if log.interventions == interventions)


def strengths(log: EvalLog, probes: list[Probe], split: Split) -> dict[str, float]:
    return {s.fact_id: s.strength for s in summarise(log, probes, split)}


def drops(base: EvalLog, log: EvalLog, probes: list[Probe], split: Split) -> dict[str, float]:
    """Drop in each fact's strength from the base model to this log."""
    before, after = strengths(base, probes, split), strengths(log, probes, split)
    return {fact_id: before[fact_id] - after[fact_id] for fact_id in before}


def best_layer(logs: list[EvalLog], fact: Fact, probes: list[Probe]) -> int:
    """The layer where erasing the fact drops it most beyond the mean drop of the other facts, on dev."""
    base = find(logs, [])

    def selectivity(log: EvalLog) -> float:
        drop = drops(base, log, probes, "dev")
        return drop[fact.id] - statistics.fmean(v for k, v in drop.items() if k != fact.id)

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


def make_windows(config: Config, smoke: bool, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Val windows (the first consecutive ones) and retain windows (random offsets in train) of
    source 0; random tokens in smoke."""
    width = config.seq_len + 1
    rng = np.random.default_rng(0)
    if smoke:
        val_tokens = train_tokens = rng.integers(16, config.vocab_size, 8 * width)
    else:
        [val_tokens] = load_tokens(config.sources[:1], "val")
        [train_tokens] = load_tokens(config.sources[:1], "train")
    val = rearrange(val_tokens[: VAL_WINDOWS * width], "(n s) -> n s", s=width)
    starts = rng.integers(0, len(train_tokens) - width, RETAIN_WINDOWS)
    retain = np.stack([train_tokens[start : start + width] for start in starts])
    return (
        torch.tensor(val.astype(np.int64), device=device),
        torch.tensor(retain.astype(np.int64), device=device),
    )


def run(config: Config, checkpoint: str, device: str, smoke: bool = False) -> list[EvalLog]:
    """Evaluate the base model, then every erasure, null, fine-tuning removal and relearning chain."""
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    facts, probes = load_probes()
    model = load_model(config, checkpoint, device, smoke)
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    val_windows, retain_windows = make_windows(config, smoke, device)
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

    def finetune_chain(
        fact: Fact,
        prefix: list[Intervention],
        make: Callable[[Fact, int], Intervention],
        retain: torch.Tensor | None,
        lr: float,
    ) -> None:
        """Fine-tune on the fact's dev probes, evaluating at each of STEPS."""
        fact_probes = [p for p in probes if p.fact_id == fact.id and p.split == "dev"]
        for step in finetune(model, tokenizer, fact_probes, retain, max(STEPS), lr, device):
            if step in STEPS:
                evaluate([*prefix, make(fact, step)])

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
    for fact in facts:
        layer = best_layer(logs, fact, probes)
        model.load_state_dict(original)
        erase(model, directions[fact.id, layer])
        finetune_chain(fact, [erasure(fact, layer)], relearn, None, RELEARN_LR)
    for fact in facts:
        model.load_state_dict(original)
        finetune_chain(fact, [], grad_diff, retain_windows, GRAD_DIFF_LR)
        finetune_chain(fact, [grad_diff(fact, max(STEPS))], relearn, None, RELEARN_LR)

    path = data.DATA_ROOT / "pilots" / "fact_removal" / f"{config.name}-{checkpoint}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(log.model_dump_json() for log in logs) + "\n")

    wandb.init(
        project="features-depth-curriculum",
        name=f"pilot-fact-removal-{config.name}-{checkpoint}",
        config={"model": config.model_dump(), "checkpoint": checkpoint, "steps": STEPS},
    )
    for name, rows in report(logs, facts, probes).items():
        print(name)
        for row in rows:
            print(row)
        table = wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])
        wandb.log({name: table})
    wandb.save(str(path), base_path=str(path.parent))
    wandb.finish()
    return logs


def report(logs: list[EvalLog], facts: list[Fact], probes: list[Probe]) -> dict[str, list[dict]]:
    """Three tables on the test split: is each fact known, how specific is its erasure, and how
    do the fine-tuning chains (relearning, GradDiff) move it."""
    base = find(logs, [])
    source = next(iter(base.val_loss))
    subject = {fact.id: fact.subject for fact in facts}

    def val_change(log: EvalLog) -> float:
        return log.val_loss[source] - base.val_loss[source]

    known, erasure_rows, finetune_rows = [], [], []
    for summary in summarise(base, probes):
        fact = next(f for f in facts if f.id == summary.fact_id)
        known.append({
            "fact": fact.id, "kind": fact.kind, "strength": summary.strength,
            "consistency": summary.consistency,
            "min_template_consistency": summary.min_template_consistency,
        })  # fmt: skip
        layer = best_layer(logs, fact, probes)
        log = find(logs, [erasure(fact, layer)])
        drop = drops(base, log, probes, "test")
        others = {k: v for k, v in drop.items() if k != fact.id}
        same_subject = [v for k, v in others.items() if subject[k] == fact.subject]
        null_drops = [drops(base, find(logs, [null(seed)]), probes, "test") for seed in NULL_SEEDS]
        erasure_rows.append({
            "fact": fact.id, "layer": layer, "target_drop": drop[fact.id],
            "same_subject_drop": statistics.fmean(same_subject) if same_subject else None,
            "other_drop": statistics.fmean(others.values()),
            "null_target_drop": statistics.fmean(d[fact.id] for d in null_drops),
            "val_loss_change": val_change(log),
        })  # fmt: skip
        for k in STEPS:
            chains = {
                "erase+relearn": [erasure(fact, layer), relearn(fact, k)],
                "grad_diff": [grad_diff(fact, k)],
                "grad_diff+relearn": [grad_diff(fact, max(STEPS)), relearn(fact, k)],
            }
            for chain, interventions in chains.items():
                log = find(logs, interventions)
                after = strengths(log, probes, "test")
                finetune_rows.append({
                    "fact": fact.id, "chain": chain, "steps": k, "target_strength": after[fact.id],
                    "other_strength": statistics.fmean(v for f, v in after.items() if f != fact.id),
                    "val_loss_change": val_change(log),
                })  # fmt: skip
    return {"known": known, "erasure": erasure_rows, "finetune": finetune_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    run(config.smoke() if args.smoke else config, args.checkpoint, args.device, args.smoke)
