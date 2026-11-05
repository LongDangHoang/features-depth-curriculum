import numpy as np
from jaxtyping import Float

from .config import Config

MUON_MOMENTUM_START, MUON_MOMENTUM_END = 0.85, 0.95  # modded-nanogpt's values


def warmup_fraction(config: Config, step: int) -> float:
    """How far through the warmup a step is, from near 0 to 1."""
    return min(1, (step + 1) * config.tokens_per_step / config.warmup_tokens)


def decay_fraction(config: Config, step: int) -> float:
    """Remaining lr multiplier from the decay: 1 before it, falling linearly toward 0 during it."""
    return min(1, (config.token_budget - step * config.tokens_per_step) / config.decay_tokens)


def lr_at(config: Config, step: int) -> float:
    """Learning rate at a step: linear warmup, constant, linear decay toward 0."""
    return config.lr * min(warmup_fraction(config, step), decay_fraction(config, step))


def muon_momentum_at(config: Config, step: int) -> float:
    """Muon momentum at a step: linear from start to end over the lr warmup."""
    warmup = warmup_fraction(config, step)
    return MUON_MOMENTUM_START + (MUON_MOMENTUM_END - MUON_MOMENTUM_START) * warmup


def weights_at(
    config: Config, source_tokens: list[int], step: int
) -> Float[np.ndarray, " sources"]:
    """Mixture weights at a step, in config.sources order, proportional to size within the phase:
    decay_sources during the lr decay, the other sources before it (all of them if none decay)."""
    in_decay = decay_fraction(config, step) < 1
    in_phase = [
        (key in config.decay_sources) == (in_decay and bool(config.decay_sources))
        for key in config.sources
    ]
    phase_tokens = np.array(source_tokens) * np.array(in_phase)
    return phase_tokens / phase_tokens.sum()


def weights_by_step(config: Config, source_tokens: list[int]) -> Float[np.ndarray, "steps sources"]:
    """Weights for every step of the run, shape (steps, n_sources)."""
    return np.stack([weights_at(config, source_tokens, step) for step in range(config.steps)])
