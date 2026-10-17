import functools
import hashlib

import lightning as L
import numpy as np
import torch
from jaxtyping import Float, Int
from torch.utils.data import DataLoader, IterableDataset

from . import data, schedule, tokens
from .config import Config


def load_tokens(source_keys: list[str], split: str) -> list[np.ndarray]:
    """Load the token file of each source for a split into RAM, in one network call instead of
    multiple small ones."""
    role_by_key = {source.key: source.role for source in data.load_registry()}
    return [
        np.fromfile(
            data.DATA_ROOT
            / "bin"
            / f"{key}.{split}.{tokens.TEMPLATE_BY_ROLE[role_by_key[key]]}.bin",
            dtype=np.uint16,
        )
        for key in source_keys
    ]


def draw_source_ids(
    weights_by_step: Float[np.ndarray, "steps sources"], batch_size: int, seed: int
) -> Int[np.ndarray, "steps batch"]:
    """Source of every sequence of the run, shape (steps, batch_size), from its own stream."""
    rng = np.random.default_rng([seed, 0])
    n_sources = weights_by_step.shape[1]
    return np.stack([rng.choice(n_sources, batch_size, p=weights) for weights in weights_by_step])


@functools.cache
def chunk_order(
    seed: int, source_key: str, epoch: int, n_chunks: int
) -> Int[np.ndarray, " n_chunks"]:
    """Shuffled chunk order of one source in one epoch; keyed by source key, not index."""
    key_hash = int(hashlib.sha1(source_key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([seed, 1, key_hash, epoch]).permutation(n_chunks)


def tokens_and_targets(windows: Int[np.ndarray, "... seq_plus_one"]) -> dict[str, torch.Tensor]:
    """Split seq_len + 1 token windows into model inputs and next-token targets, seq_len each."""
    window_tensor = torch.from_numpy(windows.astype(np.int64))
    return {"tokens": window_tensor[..., :-1], "targets": window_tensor[..., 1:]}


class Mixture:
    """Batches as a pure function of (seed, step); chunks drawn without replacement per epoch.

    The source of every sequence of the run is drawn up front (source_ids, shape (steps, batch)).
    Reading those sequences in order, each gets a draw number: 0 the first time its source comes
    up, 1 the second time, and so on, e.g. sources [A, B, A, A] get [0, 0, 1, 2].

    Each source's train tokens are cut into consecutive seq_len-token chunks; a window is a chunk
    plus the next token, its last target. Draw number d reads chunk order[d % n_chunks] of epoch
    d // n_chunks, where order is a shuffle per (source, epoch). So chunks are drawn without
    replacement within an epoch, batches depend only on (seed, step), and resume needs no loader
    state.
    """

    def __init__(
        self,
        train_tokens: list[np.ndarray],
        source_keys: list[str],
        source_ids: Int[np.ndarray, "steps batch"],
        seq_len: int,
        seed: int,
    ):
        self.train_tokens = train_tokens
        self.source_keys = source_keys
        self.source_ids = source_ids
        self.seq_len = seq_len
        self.seed = seed
        self.n_chunks = [(len(source) - 1) // seq_len for source in train_tokens]
        flat_source_ids = source_ids.ravel()  # (steps * batch,), in step order
        flat_draw_number = np.zeros_like(flat_source_ids)
        for source_index in range(len(source_keys)):
            is_source = flat_source_ids == source_index
            flat_draw_number[is_source] = np.arange(is_source.sum())
        self.draw_number: Int[np.ndarray, "steps batch"] = flat_draw_number.reshape(
            source_ids.shape
        )

    def window(self, source_index: int, draw_number: int) -> Int[np.ndarray, " window"]:
        """Tokens of the chunk that the draw_number-th draw of a source reads."""
        n_chunks = self.n_chunks[source_index]
        epoch, position = divmod(draw_number, n_chunks)
        order = chunk_order(self.seed, self.source_keys[source_index], epoch, n_chunks)
        start = int(order[position]) * self.seq_len
        return self.train_tokens[source_index][start : start + self.seq_len + 1]

    def batch(self, step: int) -> dict[str, torch.Tensor]:
        windows = [
            self.window(source_index, draw_number)
            for source_index, draw_number in zip(self.source_ids[step], self.draw_number[step])
        ]
        return {
            **tokens_and_targets(np.stack(windows)),  # each (batch, seq_len)
            "source_id": torch.from_numpy(self.source_ids[step].astype(np.int64)),
        }


class StepBatches(IterableDataset):
    """Batches from start_step to the end of the run."""

    def __init__(self, mixture: Mixture, start_step: int):
        self.mixture = mixture
        self.start_step = start_step

    def __iter__(self):
        for step in range(self.start_step, len(self.mixture.source_ids)):
            yield self.mixture.batch(step)


def val_windows(
    val_tokens: list[np.ndarray], seq_len: int, windows_per_source: int
) -> list[dict[str, torch.Tensor]]:
    """Fixed, evenly spaced validation windows per source; the same at every eval."""
    windows, source_ids = [], []
    for source_index, source in enumerate(val_tokens):
        starts = np.linspace(0, len(source) - seq_len - 1, windows_per_source).astype(np.int64)
        windows += [source[start : start + seq_len + 1] for start in starts]
        source_ids += [source_index] * windows_per_source
    return [
        {**tokens_and_targets(window), "source_id": torch.tensor(source_id)}
        for window, source_id in zip(windows, source_ids)
    ]


class MixtureData(L.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def setup(self, stage: str) -> None:
        train_tokens = load_tokens(self.config.sources, "train")
        source_tokens = [len(source) for source in train_tokens]
        weights = schedule.weights_by_step(self.config, source_tokens)
        source_ids = draw_source_ids(weights, self.config.batch_size, self.config.seed)
        self.weights = weights
        self.mixture = Mixture(
            train_tokens, self.config.sources, source_ids, self.config.seq_len, self.config.seed
        )
        val_tokens = load_tokens(self.config.sources, "val")
        self.val_dataset = val_windows(val_tokens, self.config.seq_len, self.config.val_windows)

    def train_dataloader(self) -> DataLoader:
        # Batches start at global_step, so resume needs no loader state
        return DataLoader(
            StepBatches(self.mixture, self.trainer.global_step),  # pyright: ignore[reportOptionalMemberAccess]
            batch_size=None,
            num_workers=0,  # reading from RAM is fast; multi-worker complexity deferred until needed
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)  # pyright: ignore[reportArgumentType]
