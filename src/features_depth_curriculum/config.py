from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import BaseModel, Field, model_validator

from .study.schedules import ScheduleName

# Allowlist; smoke uses the first. L4 is the smallest GPU with bf16 (T4 has none, so bf16 +
# compile would fail there). These bounds are guardrails, not a security control: the agent
# can edit this file. The Modal workspace spend limit is the backstop.
Gpu = Literal["L4", "T4", "A10G", "A100", "H100"]
GPUS: tuple[str, ...] = get_args(Gpu)


FactType = Literal["natural", "abstract", "random", "trivia", "foundational"]
FACTS_DIR = Path(__file__).parent / "study" / "facts"


class Injection(BaseModel, extra="forbid"):
    """Made-up facts to mix into training: `exposures` readings of each fact's texts in all,
    spread over the run by the schedule. Name the facts by id, or by `fact_type` for all of a type.
    The texts are in study/facts/<facts_file>_texts.yml (see injection.py)."""

    facts_file: str = "made_up"
    fact_ids: list[str] = []
    fact_type: FactType | None = None
    schedule: ScheduleName
    exposures: int = Field(gt=0)

    def selected_fact_ids(self) -> list[str]:
        """The listed fact ids, or every fact of the fact type, in file order."""
        spec = yaml.safe_load((FACTS_DIR / f"{self.facts_file}.yml").read_text(encoding="utf-8"))
        known = {fact["id"]: fact["fact_type"] for fact in spec["facts"]}
        if self.fact_ids:
            return self.fact_ids
        return [fact_id for fact_id, fact_type in known.items() if fact_type == self.fact_type]

    @model_validator(mode="after")
    def _facts_exist(self) -> Injection:
        """Fail when the config loads, not on the GPU: typos, empty selections, missing texts."""
        if bool(self.fact_ids) == (self.fact_type is not None):
            raise ValueError("give either fact_ids or fact_type")
        selected = self.selected_fact_ids()
        texts_path = FACTS_DIR / f"{self.facts_file}_texts.yml"
        with_texts = yaml.safe_load(texts_path.read_text(encoding="utf-8"))
        missing = [fact_id for fact_id in selected if fact_id not in with_texts]
        if not selected or missing:
            raise ValueError(f"no facts selected, or facts without texts: {missing}")
        return self


class Config(BaseModel, extra="forbid"):
    name: str
    gpu: Gpu = "L4"
    timeout_s: int = Field(default=3600, ge=60, le=4 * 3600)
    max_containers: int = Field(default=1, ge=1, le=4)
    seed: int = 0
    sources: list[str]
    injections: list[Injection] = []  # none: the run trains on `sources` alone
    depth: int = 8
    width: int = 512
    heads: int = 8
    seq_len: int = 512
    vocab_size: int = 8192
    batch_size: int = 128
    token_budget: int
    warmup_tokens: int
    decay_tokens: int
    eval_interval_tokens: int
    lr: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_windows: int = 256  # per source
    sample_tokens: int = 64
    # GPU-only features: off in local CPU smoke, on in the Modal GPU smoke
    # Modded-nanogpt recipe: Muon on block matrices, ReLU^2 MLP, QK-norm, zero-init output projections
    speedrun: bool = False
    bf16: bool = False
    compile: bool = False

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def steps(self) -> int:
        return self.token_budget // self.tokens_per_step

    @property
    def eval_interval_steps(self) -> int:
        return self.eval_interval_tokens // self.tokens_per_step

    def smoke(self, keep_gpu_features: bool = False) -> Config:
        seq_len, batch_size = 32, 4
        tokens_per_step = seq_len * batch_size
        return self.model_copy(
            update={
                "name": f"{self.name}-smoke-{time.strftime('%Y%m%d-%H%M%S')}",
                "depth": 1,
                "width": 32,
                "heads": 2,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "token_budget": 4 * tokens_per_step,
                "warmup_tokens": tokens_per_step,
                "decay_tokens": 2 * tokens_per_step,
                "eval_interval_tokens": 2 * tokens_per_step,
                "val_windows": 2,
                "sample_tokens": 4,
                "bf16": self.bf16 and keep_gpu_features,
                "compile": self.compile and keep_gpu_features,
            }
        )


def load(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))
