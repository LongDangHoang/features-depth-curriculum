"""C: scenarios, how the facts got into the model (design.md, Scenarios).

`checkpoint` loads a trained model. The injection schedules, `baseline` and `seed-twin` are runs
made by the trainer (see Config.injections and injection.py), so each one is `checkpoint` with
its own experiment config. The three scenarios below start from a trained model and add the
made-up facts afterwards: `rank1` by editing one weight matrix, `finetune` and `dose-ladder` by
training. They add every fact of the fact file, one after the other or all together.
"""

import warnings
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from jaxtyping import Float, Int
from tokenizers import Tokenizer
from torch import nn

from ..injection import MAX_INJECTED_SHARE, build_windows, fact_texts, pack_texts
from ..mixture import load_tokens
from ..model import GPT, Block
from ..pilot.run import load_model
from ..pilot.scoring import encode_prompts, token_id
from .interface import Context, Fact, Probe, Scenario, load_facts, register

SMOKE_STEPS = 3  # in smoke, every loop stops after this many steps


def encoded_margins(
    model: GPT,
    tokenizer: Tokenizer,
    probes: list[Probe],
    token_ids: Int[torch.Tensor, "probes seq"],
    last_index: Int[torch.Tensor, " probes"],
) -> Float[torch.Tensor, " probes"]:
    """For each (encoded) probe, the log-prob of the target minus that of the best foil, in nats."""
    rows = torch.arange(len(probes), device=token_ids.device)
    logprobs = model(token_ids)[rows, last_index].float().log_softmax(dim=-1)  # (probes, vocab)
    return torch.stack(
        [
            logprobs[row, token_id(tokenizer, probe.target)]
            - max(logprobs[row, token_id(tokenizer, foil)] for foil in probe.foils)
            for row, probe in enumerate(probes)
        ]
    )


@torch.no_grad()
def mean_margin(model: GPT, probes: list[Probe], context: Context) -> float:
    token_ids, last_index = encode_prompts(
        context.tokenizer, [probe.prompt for probe in probes], context.device
    )
    return encoded_margins(model, context.tokenizer, probes, token_ids, last_index).mean().item()


def random_windows(
    stream: Int[np.ndarray, " tokens"],
    count: int,
    seq_len: int,
    rng: np.random.Generator,
    device: str,
) -> Int[torch.Tensor, "count seq_plus_one"]:
    """Windows of seq_len + 1 tokens that start at random places of a token stream."""
    starts = rng.integers(0, len(stream) - seq_len, count)
    windows = np.array([stream[start : start + seq_len + 1] for start in starts])
    return torch.tensor(windows.astype(np.int64), device=device).reshape(count, seq_len + 1)


def corpus_windows(
    streams: list[Int[np.ndarray, " tokens"]],
    count: int,
    seq_len: int,
    rng: np.random.Generator,
    device: str,
) -> Int[torch.Tensor, "count seq_plus_one"]:
    """Random windows from all corpus sources, in proportion to their size."""
    sizes = np.array([len(stream) for stream in streams])
    picks = rng.choice(len(streams), count, p=sizes / sizes.sum())
    return torch.cat(
        [
            random_windows(stream, int((picks == index).sum()), seq_len, rng, device)
            for index, stream in enumerate(streams)
        ]
    )


def train_step(
    model: GPT, optimizer: torch.optim.Optimizer, windows: Int[torch.Tensor, "batch seq_plus_one"]
) -> None:
    """One step of next-token training on windows."""
    logits = model(windows[:, :-1])  # (batch, seq, vocab)
    loss = F.cross_entropy(
        rearrange(logits.float(), "batch seq vocab -> (batch seq) vocab"),
        rearrange(windows[:, 1:], "batch seq -> (batch seq)"),
    )
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def short_in_smoke(steps: int, context: Context) -> int:
    return min(steps, SMOKE_STEPS) if context.smoke else steps


class AddedFacts(Scenario):
    """Base of the scenarios that start from a trained model and add the made-up facts."""

    facts_file: str = "made_up"  # the fact file whose facts, and their texts, are added

    def build(self, context: Context) -> GPT:
        model = load_model(context.config, self.checkpoint, context.device, context.smoke)
        facts, probes = load_facts([self.facts_file])
        self.add_facts(model, facts, probes, fact_texts(self.facts_file), context)
        return model.eval()

    def add_facts(
        self,
        model: GPT,
        facts: list[Fact],
        probes: list[Probe],
        texts_by_fact: dict[str, list[str]],
        context: Context,
    ) -> None:
        raise NotImplementedError


@register
class Rank1(AddedFacts):
    """A planted fact: a rank-one edit of one MLP output matrix, with no training (ROME, simplified).

    Per fact:
      key    the mean input of the matrix at the last token of the subject, over the fact's texts;
      delta  the change of the matrix output at that key, found by gradient steps under the real
             edit (every position gets delta * (input . key) / (key . key)) until the answer
             beats the best foil by `target_margin` nats on the fact's dev prompts.
    Then all facts are edited at once, W += delta_matrix (K^T K)^-1 K^T, with the keys as the
    columns of K, so that the edit of one fact does not move another fact at its key.
    Each fact's dev margin after the edit is printed, with a warning if it is below the target.

    `target_margin` is the strength K. Set it to what the injected facts reach, so that rank1 is
    compared with them at the same strength. (With the 1-layer smoke model the edit changes
    nothing: no later layer reads it. Smoke only checks that the code runs.)
    """

    name: Literal["rank1"] = "rank1"
    layer: int = 6  # the block whose MLP output matrix is edited
    target_margin: float = 2.0
    max_steps: int = 200  # gradient steps to find delta, per fact
    learning_rate: float = 0.05

    def add_facts(self, model, facts, probes, texts_by_fact, context):
        block = model.blocks[min(self.layer, len(model.blocks) - 1)]
        assert isinstance(block, Block)
        down = block.mlp.down
        dev_probes = {
            fact.id: [p for p in probes if p.fact_id == fact.id and p.split == "dev"]
            for fact in facts
        }
        model.requires_grad_(False)  # only delta learns
        keys = [
            self.key(model, down, fact.subject, texts_by_fact[fact.id], context) for fact in facts
        ]
        deltas = [
            self.delta(model, down, key, dev_probes[fact.id], context)
            for fact, key in zip(facts, keys)
        ]
        key_matrix = torch.stack(keys, dim=1)  # (hidden, facts)
        delta_matrix = torch.stack(deltas, dim=1)  # (width, facts)
        down.weight.data += delta_matrix @ torch.linalg.pinv(key_matrix)  # (width, hidden)
        model.requires_grad_(True)
        for fact in facts:
            reached = mean_margin(model, dev_probes[fact.id], context)
            print(f"rank1 {fact.id}: dev margin {reached:.2f} (target {self.target_margin})")
            if reached < self.target_margin:
                warnings.warn(f"rank1 {fact.id} reached {reached:.2f} < {self.target_margin}")

    def key(
        self, model: GPT, down: nn.Linear, subject: str, texts: list[str], context: Context
    ) -> Float[torch.Tensor, " hidden"]:
        """The mean input of the matrix at the last token of the subject, over the texts."""
        prefixes = [text[: end_of_subject(text, subject)] for text in texts]
        token_ids, last_index = encode_prompts(context.tokenizer, prefixes, context.device)
        inputs: list[torch.Tensor] = []
        hook = down.register_forward_hook(lambda module, args, output: inputs.append(args[0]))
        with torch.no_grad():
            model(token_ids)
        hook.remove()
        rows = torch.arange(len(texts), device=context.device)
        return reduce(inputs[0][rows, last_index], "text hidden -> hidden", "mean")

    def delta(
        self,
        model: GPT,
        down: nn.Linear,
        key: Float[torch.Tensor, " hidden"],
        probes: list[Probe],
        context: Context,
    ) -> Float[torch.Tensor, " width"]:
        """The output change at the key that makes the answer win by target_margin on the probes.
        The hook computes what the edited matrix would: output + delta * (input . key) / (key . key)."""
        token_ids, last_index = encode_prompts(
            context.tokenizer, [probe.prompt for probe in probes], context.device
        )
        delta = torch.zeros(down.out_features, device=context.device, requires_grad=True)
        hook = down.register_forward_hook(
            lambda module, args, output: (
                output + rearrange(args[0] @ key / key.dot(key), "b s -> b s 1") * delta
            )
        )
        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)
        for _ in range(short_in_smoke(self.max_steps, context)):
            margin = encoded_margins(model, context.tokenizer, probes, token_ids, last_index).mean()
            if margin.item() >= self.target_margin:
                break
            optimizer.zero_grad()
            (-margin).backward()
            optimizer.step()
        hook.remove()
        return delta.detach()


def end_of_subject(text: str, subject: str) -> int:
    """The position in the text just after the first mention of the subject (any case)."""
    return text.lower().index(subject.lower()) + len(subject)


@register
class Finetune(AddedFacts):
    """Late fine-tune on the facts' texts, until the mean margin on the dev prompts reaches
    `target_margin` or `max_steps` steps are done. (The test prompts are for reporting.)
    Set `target_margin` to what the injected facts reach."""

    name: Literal["finetune"] = "finetune"
    target_margin: float = 2.0
    max_steps: int = 500
    batch_size: int = 16
    learning_rate: float = 1e-4
    check_every: int = 10  # steps between two margin checks

    def add_facts(self, model, facts, probes, texts_by_fact, context):
        texts = [text for fact in facts for text in texts_by_fact[fact.id]]
        stream = pack_texts(texts, context.tokenizer, context.config.seed)
        dev_probes = [probe for probe in probes if probe.split == "dev"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(context.config.seed)
        for step in range(short_in_smoke(self.max_steps, context)):
            checking = step % self.check_every == 0
            if checking and mean_margin(model, dev_probes, context) >= self.target_margin:
                break
            windows = random_windows(
                stream, self.batch_size, context.config.seq_len, rng, context.device
            )
            train_step(model, optimizer, windows)


@register
class DoseLadder(AddedFacts):
    """Continued training on corpus windows, with the facts' texts mixed in so that each fact is
    seen `exposures` times in `tokens` tokens. One value of `exposures` is one rung of the ladder.
    The fact windows are spread evenly over the steps. It is an error if they need more than
    MAX_INJECTED_SHARE of the sequences."""

    name: Literal["dose-ladder"] = "dose-ladder"
    exposures: int = 100
    tokens: int = 4_000_000
    batch_size: int = 16
    learning_rate: float = 1e-4

    def add_facts(self, model, facts, probes, texts_by_fact, context):
        config = context.config
        if context.smoke:
            facts = facts[:2]  # a small dose: smoke has only a few steps
        exposures = 1 if context.smoke else self.exposures
        texts = [text for fact in facts for text in texts_by_fact[fact.id]]
        fact_windows = build_windows(
            texts, exposures, len(facts), context.tokenizer, config.seq_len, [config.seed, 3]
        )
        if context.smoke:
            corpus = [np.random.default_rng(0).integers(16, config.vocab_size, 10 * config.seq_len)]
        else:
            corpus = load_tokens(config.sources, "train")
        steps = short_in_smoke(self.tokens // (self.batch_size * config.seq_len), context)
        share = len(fact_windows) / (steps * self.batch_size)
        if share > MAX_INJECTED_SHARE:
            raise ValueError(
                f"{exposures} exposures need {share:.0%} of the sequences to be fact windows "
                f"(at most {MAX_INJECTED_SHARE:.0%}): lower exposures or raise tokens"
            )
        fact_windows_before_step = np.round(
            np.arange(steps + 1) * len(fact_windows) / steps
        ).astype(int)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(config.seed)
        for step in range(steps):
            from_facts = fact_windows[
                fact_windows_before_step[step] : fact_windows_before_step[step + 1]
            ]
            windows = torch.cat(
                [
                    torch.tensor(from_facts.astype(np.int64), device=context.device),
                    corpus_windows(
                        corpus,
                        self.batch_size - len(from_facts),
                        config.seq_len,
                        rng,
                        context.device,
                    ),
                ]
            )  # (batch, seq + 1)
            train_step(model, optimizer, windows)


@register
class Checkpoint(Scenario):
    """A model that is already trained: `baseline`, `seed-twin` and every injection schedule are
    trained runs, so each one is this scenario with its own experiment config."""

    name: Literal["checkpoint"] = "checkpoint"

    def build(self, context: Context) -> GPT:
        return load_model(context.config, self.checkpoint, context.device, context.smoke)
