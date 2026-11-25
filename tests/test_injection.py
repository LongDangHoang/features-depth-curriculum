import numpy as np
from tokenizers import Tokenizer

from features_depth_curriculum import data, injection, mixture
from features_depth_curriculum.config import Injection, load
from features_depth_curriculum.study.schedules import SCHEDULES, sequence_steps

SEQ_LEN, BATCH_SIZE, STEPS = 16, 8, 50


def test_every_schedule_has_the_same_total_and_places_sequences_exactly():
    fractions = (np.arange(20_000) + 0.5) / 20_000  # the middle of each slice of training
    totals = {name: shape(fractions).mean() for name, shape in SCHEDULES.items()}
    assert np.allclose(list(totals.values()), 1.0, atol=1e-3), totals
    steps = sequence_steps("early-light", 100, 1000)
    assert len(steps) == 100 and steps.max() < 300  # all early; the count is exact, not sampled


def test_injections_overwrite_slots_and_leave_other_corpus_draws_alone():
    train_tokens = [np.arange(SEQ_LEN * 40 + 1, dtype=np.uint16)]
    corpus_ids = np.zeros((STEPS, BATCH_SIZE), dtype=np.int64)
    plain = mixture.Mixture(train_tokens, ["corpus"], corpus_ids, SEQ_LEN, seed=0)
    config = load("experiments/sanity.yml").model_copy(
        update={"injections": [Injection(fact_ids=["zavv_bird"], schedule="uniform", exposures=10)]}
    )
    tokenizer = Tokenizer.from_file(str(data.REPO_ROOT / "tokenizer.json"))
    windows = injection.build_injection(config.injections[0], tokenizer, SEQ_LEN, 0, 0)
    steps = sequence_steps("uniform", len(windows), STEPS)
    injected = mixture.Mixture(
        train_tokens, ["corpus", "injection"], corpus_ids, SEQ_LEN, 0, [windows], [steps]
    )
    overwritten = injected.source_ids == 1
    assert overwritten.sum() == len(windows)
    assert np.array_equal(injected.draw_number[~overwritten], plain.draw_number[~overwritten])
    assert injection.planned_exposures(config)["zavv_bird"]["texts"] > 0
