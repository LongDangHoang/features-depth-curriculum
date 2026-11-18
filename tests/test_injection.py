import numpy as np

from features_depth_curriculum import schedule
from features_depth_curriculum.config import Injection, load
from features_depth_curriculum.study.schedules import SCHEDULES

SOURCE_TOKENS = [300, 100]


def test_every_schedule_has_the_same_total():
    fractions = (np.arange(20_000) + 0.5) / 20_000  # the middle of each slice of training
    totals = {name: shape(fractions).mean() for name, shape in SCHEDULES.items()}
    assert np.allclose(list(totals.values()), 1.0, atol=1e-3), totals


def test_no_injections_keeps_the_corpus_proportions_and_injections_add_sources():
    config = load("experiments/sanity.yml").model_copy(update={"sources": ["a", "b"]})
    assert np.array_equal(
        schedule.weights_by_step(config, SOURCE_TOKENS),
        np.tile([0.75, 0.25], (config.steps, 1)),
    )
    injection = Injection(fact_ids=["zib_bird"], schedule="early-light", exposures=10)
    with_injection = config.model_copy(update={"injections": [injection]})
    weights = schedule.weights_by_step(with_injection, SOURCE_TOKENS, injection_draws=[1000.0])
    assert weights.shape == (config.steps, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.isclose(weights[:, 2].sum(), 1000.0 / config.batch_size, rtol=1e-3)  # draws / batch
