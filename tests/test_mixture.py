import numpy as np
import torch

from features_depth_curriculum import mixture

SEQ_LEN = 16
BATCH_SIZE = 8
STEPS = 50
SEED = 0


def build_mixture() -> mixture.Mixture:
    train_tokens = [
        np.arange(SEQ_LEN * 40 + 1, dtype=np.uint16),
        np.arange(SEQ_LEN * 5 + 1, dtype=np.uint16),
    ]
    weights_by_step = np.tile([0.3, 0.7], (STEPS, 1))
    source_ids = mixture.draw_source_ids(weights_by_step, BATCH_SIZE, SEED)
    return mixture.Mixture(train_tokens, ["large", "small"], source_ids, SEQ_LEN, SEED)


def test_batches_are_a_pure_function_of_seed_and_step():
    first, second = build_mixture(), build_mixture()
    for step in (0, 1, 7, 40):
        for key in ("tokens", "targets", "source_id"):
            assert torch.equal(first.batch(step)[key], second.batch(step)[key])
    resumed_batch = next(iter(mixture.StepBatches(first, 3)))
    assert torch.equal(resumed_batch["tokens"], first.batch(3)["tokens"])
    assert resumed_batch["tokens"].shape == resumed_batch["targets"].shape == (BATCH_SIZE, SEQ_LEN)
    assert torch.equal(resumed_batch["tokens"][:, 1:], resumed_batch["targets"][:, :-1])
    assert resumed_batch["source_id"].shape == (BATCH_SIZE,)


def test_chunks_are_drawn_without_replacement_within_an_epoch():
    built = build_mixture()
    small_source = 1
    n_chunks = built.n_chunks[small_source]
    draws = int((built.source_ids == small_source).sum())
    assert draws >= n_chunks
    first_epoch = {built.window(small_source, draw).tobytes() for draw in range(n_chunks)}
    assert len(first_epoch) == n_chunks
