import numpy as np
from jaxtyping import Float

from .config import Config


def lr_at(config: Config, step: int) -> float:
    """Learning rate at a step: linear warmup, constant, linear decay toward 0."""
    warmup = min(1, (step + 1) * config.tokens_per_step / config.warmup_tokens)
    decay = min(1, (config.token_budget - step * config.tokens_per_step) / config.decay_tokens)
    return config.lr * min(warmup, decay)


def weights_at(
    config: Config, source_tokens: list[int], step: int
) -> Float[np.ndarray, " sources"]:
    """Mixture weights at a step, in config.sources order; constant and proportional to size."""
    return np.array(source_tokens) / sum(source_tokens)


def weights_by_step(config: Config, source_tokens: list[int]) -> Float[np.ndarray, "steps sources"]:
    """Weights for every step of the run, shape (steps, n_sources)."""
    return np.stack([weights_at(config, source_tokens, step) for step in range(config.steps)])
