from collections.abc import Sequence

import numpy as np
from jaxtyping import Float

from .config import Config
from .injection import MAX_INJECTED_SHARE
from .study.schedules import SCHEDULES


def lr_at(config: Config, step: int) -> float:
    """Learning rate at a step: linear warmup, constant, linear decay toward 0."""
    warmup = min(1, (step + 1) * config.tokens_per_step / config.warmup_tokens)
    decay = min(1, (config.token_budget - step * config.tokens_per_step) / config.decay_tokens)
    return config.lr * min(warmup, decay)


def weights_at(
    config: Config, source_tokens: list[int], step: int, injection_draws: Sequence[float] = ()
) -> Float[np.ndarray, " sources"]:
    """Mixture weights at a step: config.sources, then the injections of config.injections.

    The corpus sources share what the injections leave, in proportion to their size. An injection
    gets its draws over the run (see injection.py) times its schedule's density at this step,
    divided by the sequences of the whole run. With no injections this is the corpus proportions.
    """
    fraction = np.array([(step + 0.5) / config.steps])  # the middle of the step
    sequences_in_run = config.steps * config.batch_size
    injected = np.array(
        [
            draws * SCHEDULES[injection.schedule](fraction)[0] / sequences_in_run
            for injection, draws in zip(config.injections, injection_draws, strict=True)
        ]
    )
    injected = np.minimum(injected, MAX_INJECTED_SHARE / max(len(injected), 1))
    corpus = np.array(source_tokens) / sum(source_tokens)
    return np.concatenate([corpus * (1 - injected.sum()), injected])


def weights_by_step(
    config: Config, source_tokens: list[int], injection_draws: Sequence[float] = ()
) -> Float[np.ndarray, "steps sources"]:
    """Weights for every step of the run, shape (steps, n_sources)."""
    return np.stack(
        [weights_at(config, source_tokens, step, injection_draws) for step in range(config.steps)]
    )
