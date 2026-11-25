"""Injection schedules: how a fact's exposures are spread over training (design.md, Scenarios).

Each schedule is a density over the fraction of training, from 0 (first step) to 1 (last step).
The density has an average of 1 over the run, so every schedule has the same total. `sequence_steps`
turns a schedule into exact steps for a given number of sequences, so the dose is not random. The shape is a bump (a Gaussian) plus, for the heavy schedules, a tail that
decays slowly from the peak to the end of the run.

The learning rate starts to decay at 80% of the run in the stories60m configs, so every peak
comes before that. Light schedules have no tail: they stop soon after the peak.
"""

from typing import Literal

import numpy as np
from jaxtyping import Float, Int

ScheduleName = Literal[
    "uniform", "early-heavy", "early-light", "mid-normal", "late-heavy", "late-light"
]

EARLY_PEAK = 0.10
MIDDLE_PEAK = 0.40
LATE_PEAK = 0.65
BUMP_WIDTH = 0.04  # standard deviation of the bump, as a fraction of training
TAIL_HEIGHT = 0.3  # the tail starts at this share of the peak height
TAIL_DECAY = 0.3  # the tail falls by a factor e every this fraction of training
GRID_POINTS = 10_000  # to find the average of a shape over the run

Fractions = Float[np.ndarray, " points"]


def density(fraction: Fractions, peak: float, heavy_tail: bool) -> Fractions:
    """A bump at `peak`, with an optional tail after it, scaled to an average of 1 over 0..1."""

    def shape(at: Fractions) -> Fractions:
        bump = np.exp(-0.5 * ((at - peak) / BUMP_WIDTH) ** 2)
        tail = TAIL_HEIGHT * np.exp(-(at - peak) / TAIL_DECAY) * (at > peak)
        return bump + tail if heavy_tail else bump

    grid = (np.arange(GRID_POINTS) + 0.5) / GRID_POINTS  # (GRID_POINTS,), the middle of each slice
    return shape(fraction) / shape(grid).mean()


def uniform(fraction: Fractions) -> Fractions:
    return np.ones_like(fraction)


def early_heavy(fraction: Fractions) -> Fractions:
    return density(fraction, EARLY_PEAK, heavy_tail=True)


def early_light(fraction: Fractions) -> Fractions:
    return density(fraction, EARLY_PEAK, heavy_tail=False)


def mid_normal(fraction: Fractions) -> Fractions:
    return density(fraction, MIDDLE_PEAK, heavy_tail=False)


def late_heavy(fraction: Fractions) -> Fractions:
    return density(fraction, LATE_PEAK, heavy_tail=True)


def late_light(fraction: Fractions) -> Fractions:
    return density(fraction, LATE_PEAK, heavy_tail=False)


SCHEDULES = {
    "uniform": uniform,
    "early-heavy": early_heavy,
    "early-light": early_light,
    "mid-normal": mid_normal,
    "late-heavy": late_heavy,
    "late-light": late_light,
}


def sequence_steps(schedule: ScheduleName, n_sequences: int, n_steps: int) -> Int[np.ndarray, " n"]:
    """The step of each of n sequences, so that they follow the schedule exactly: the k-th
    sequence is at the step where the cumulative density reaches (k + 0.5) / n."""
    fractions = (np.arange(n_steps) + 0.5) / n_steps  # (n_steps,), the middle of each step
    cumulative = np.cumsum(SCHEDULES[schedule](fractions))
    quantiles = (np.arange(n_sequences) + 0.5) / n_sequences * cumulative[-1]
    return np.searchsorted(cumulative, quantiles)
