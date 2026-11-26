"""A: metrics, one number per fact (metrics.md, Metrics).

Values are as measured, never sign-flipped: which direction reads as more robust depends on
the intervention (metrics.md, Valid and invalid pairs). Each metric reads the test probes of
the subject it is given, so the `prompt-story` and `prompt-wiki` interventions (which keep only
the probes of one style) change what is read. Nonce probes are the same templates with a
made-up subject: the model with no fact in it.
"""

import copy
import re
from collections import defaultdict
from statistics import mean
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float, Int
from pydantic import Field
from tokenizers import Tokenizer

from ..callbacks import END_OF_TEXT_ID
from ..mixture import load_tokens
from ..model import GPT
from ..pilot.scoring import encode_prompts, residuals, token_id
from .interface import Context, Metric, Probe, Reading, Subject, register

CHUNK = 256  # prompts per forward pass


def test_probes(subject: Subject) -> list[Probe]:
    """The next-token test probes: the ones read through log-probs."""
    return [
        probe for probe in subject.probes if probe.kind == "next-token" and probe.split == "test"
    ]


def real_minus_nonce(
    probes: list[Probe], values: Float[torch.Tensor, " probes"], context: Context
) -> dict[str, float]:
    """For each fact: for each template, the mean value on its probes minus the mean on its
    nonce probes; then the mean of that over the templates. A fact with no probes gets no entry."""
    values_by_group: dict[tuple[str, str, bool], list[float]] = defaultdict(list)
    templates_of_fact: dict[str, set[str]] = defaultdict(set)
    for probe, value in zip(probes, values.tolist()):
        values_by_group[probe.fact_id, probe.template, probe.nonce].append(value)
        if not probe.nonce:
            templates_of_fact[probe.fact_id].add(probe.template)
    return {
        fact.id: mean(
            mean(values_by_group[fact.id, template, False])
            - mean(values_by_group[fact.id, template, True])
            for template in templates_of_fact[fact.id]
        )
        for fact in context.facts
        if fact.id in templates_of_fact
    }


@torch.no_grad()
def next_token_logprobs(
    model: GPT, probes: list[Probe], context: Context
) -> tuple[Float[torch.Tensor, "prompts vocab"], dict[str, int]]:
    """Log-probabilities of the next token after each distinct prompt of the probes, and the
    row of each prompt. Each distinct prompt goes through the model once (nonce prompts repeat
    across facts)."""
    model.eval()
    prompts = list(dict.fromkeys(probe.prompt for probe in probes))
    chunks = []
    for start in range(0, len(prompts), CHUNK):
        token_ids, last_index = encode_prompts(
            context.tokenizer, prompts[start : start + CHUNK], context.device
        )
        rows = torch.arange(len(token_ids), device=context.device)
        chunks.append(F.log_softmax(model(token_ids)[rows, last_index].float(), dim=-1))
    return torch.cat(chunks), {prompt: row for row, prompt in enumerate(prompts)}


def answer_margins(
    model: GPT, probes: list[Probe], context: Context
) -> Float[torch.Tensor, " probes"]:
    """m_a = log p(answer) - logsumexp(log p(foils)) for the next token of each probe."""
    logprobs, row_of = next_token_logprobs(model, probes, context)
    margins = []
    for probe in probes:
        row = row_of[probe.prompt]
        foil_ids = [token_id(context.tokenizer, foil) for foil in probe.foils]
        answer_logprob = logprobs[row, token_id(context.tokenizer, probe.target)]
        margins.append(answer_logprob - torch.logsumexp(logprobs[row, foil_ids], dim=0))
    return torch.stack(margins)


def strength(model: GPT, probes: list[Probe], context: Context) -> dict[str, float]:
    """K_f in nats: the margin on the probes, minus the margin on the nonce probes."""
    return real_minus_nonce(probes, answer_margins(model, probes, context), context)


@register
class Margin(Metric):
    """K_f, the strength of the fact in nats, on the test probes of `after`. The share left
    (K after / K before) is computed in the analysis, not here."""

    name: Literal["margin"] = "margin"

    def measure(self, after: Subject, before: Subject, context: Context) -> dict[str, float]:
        return strength(after.model, test_probes(after), context)


@torch.no_grad()
def last_token_residuals(
    model: GPT, probes: list[Probe], layer: int, context: Context
) -> Float[torch.Tensor, "probes width"]:
    """The residual stream at the last token of each probe's prompt, after block `layer`."""
    model.eval()
    chunks = []
    for start in range(0, len(probes), CHUNK):
        token_ids, last_index = encode_prompts(
            context.tokenizer,
            [probe.prompt for probe in probes[start : start + CHUNK]],
            context.device,
        )
        chunks.append(residuals(model, token_ids, last_index)[layer])  # (chunk, width)
    return torch.cat(chunks)


def separation(
    projections_of_answer: Float[torch.Tensor, " answer_probes"],
    projections_of_foils: Float[torch.Tensor, " foil_probes"],
) -> float:
    """The difference of the two means, in pooled standard deviations."""
    pooled_sd = ((projections_of_answer.var() + projections_of_foils.var()) / 2).sqrt()
    if pooled_sd == 0:
        raise ValueError("presence: the statements do not vary along the direction (SD is 0)")
    return ((projections_of_answer.mean() - projections_of_foils.mean()) / pooled_sd).item()


@register
class Presence(Metric):
    """How far statements of the fact's answer separate from statements of its foils, along the
    answer's direction, in texts that never mention the fact's subject (metrics.md, Metrics,
    presence).

    The statements are `before.probes`' statement probes: they are the instrument, so a
    prompt-style intervention does not change them. For each answer, the direction is the unit
    mean difference of the last-token residuals of dev statements of the answer and of dev
    statements of the other answers of its relation. Value, `fixed`: on the test statements read
    with `after.model`, the mean projection on the direction built with `before.model` for the
    answer minus the same for the foils, in pooled standard deviations. `info` holds
    `presence_refit`: the same with the direction built on `after.model` (whether any
    representation survives). A fact's row holds the value of its answer.

    `layer` counts from 1 (after the first block) to the model depth.
    """

    name: Literal["presence"] = "presence"
    layer: int = Field(default=4, ge=1)

    def measure(
        self, after: Subject, before: Subject, context: Context
    ) -> dict[str, float | Reading]:
        depth = len(before.model.blocks)
        layer = min(self.layer, depth) if context.smoke else self.layer  # smoke has depth 1
        if layer > depth:
            raise ValueError(f"presence layer {layer} is above the model depth {depth}")
        statements = {
            split: [
                probe
                for probe in before.probes
                if probe.kind == "statement" and probe.split == split
            ]
            for split in ("dev", "test")
        }
        if not statements["dev"] or not statements["test"]:
            raise ValueError("presence needs statement probes: add `statements` to the fact file")
        relations = {
            split: np.array([probe.relation for probe in probes_of_split])
            for split, probes_of_split in statements.items()
        }
        answers = {
            split: np.array([probe.answer for probe in probes_of_split])
            for split, probes_of_split in statements.items()
        }
        # (statements, width), once per model and split
        residuals_of = {
            (model_name, split): last_token_residuals(model, statements[split], layer, context)
            for model_name, model in (("before", before.model), ("after", after.model))
            for split in statements
        }

        def read(direction_model: str, read_model: str, relation: str, answer: str) -> float:
            """Separation of the test statements of `read_model` on the direction built from
            the dev statements of `direction_model`."""
            masks = {}
            for split in statements:
                in_relation = relations[split] == relation
                masks[split] = (
                    torch.from_numpy(in_relation & (answers[split] == answer)).to(context.device),
                    torch.from_numpy(in_relation & (answers[split] != answer)).to(context.device),
                )
                if not masks[split][0].any() or not masks[split][1].any():
                    raise ValueError(
                        f"presence: no {split} statements of {answer!r} or of the other "
                        f"{relation} answers: add forms and statements to the fact file"
                    )
            dev_residuals = residuals_of[direction_model, "dev"]
            direction = F.normalize(
                dev_residuals[masks["dev"][0]].mean(dim=0)
                - dev_residuals[masks["dev"][1]].mean(dim=0),
                dim=0,
            )
            projections = residuals_of[read_model, "test"] @ direction  # (statements,)
            return separation(projections[masks["test"][0]], projections[masks["test"][1]])

        readings = {
            (relation, answer): Reading(
                value=read("before", "after", relation, answer),
                info={"presence_refit": read("after", "after", relation, answer)},
            )
            for relation, answer in dict.fromkeys(
                (fact.relation, fact.answer) for fact in context.facts
            )
        }
        return {fact.id: readings[fact.relation, fact.answer] for fact in context.facts}


@register
class JudgeElicit(Metric):
    """The share of sampled continuations that say the fact.

    The prompts are the `elicit` probes of `after`: story openings that bring up the subject
    without asking for the attribute, so a low share is expected. For each, sample `samples`
    continuations of `max_new_tokens` tokens at temperature 1. A continuation succeeds if the
    answer word comes before any foil word. A word matches whole or with a common ending, an
    optional doubled consonant first ("barks", "colder", "hopping"), so "red" does not match
    "ready". The continuation stops at the first end-of-text, where an unrelated story starts.
    A fact with no elicit probes (after `prompt-wiki`, say) gets no row.
    """

    name: Literal["judge-elicit"] = "judge-elicit"
    judge: Literal["word"] = "word"  # later: "llm", for facts that have no single word
    samples: int = Field(default=4, ge=1)
    max_new_tokens: int = Field(default=30, ge=1)
    seed: int = 0

    def says_the_answer(self, continuation: str, probe: Probe) -> bool:
        # An LLM judge plugs in here later: ask it whether the text states the fact.
        answer = probe.target.strip()
        words = [answer] + [foil.strip() for foil in probe.foils]
        alternatives = "|".join(map(re.escape, words))
        word_pattern = rf"\b({alternatives})(?:[a-z]?(?:s|es|ed|er|est|ing))?\b"
        first_match = re.search(word_pattern, continuation.lower())
        return first_match is not None and first_match.group(1) == answer

    @torch.no_grad()
    def measure(self, after: Subject, before: Subject, context: Context) -> dict[str, float]:
        after.model.eval()
        generator = torch.Generator(device=context.device).manual_seed(self.seed)
        successes: dict[str, list[bool]] = defaultdict(list)
        for probe in after.probes:
            if probe.kind != "elicit":
                continue
            prompt_ids, _ = encode_prompts(context.tokenizer, [probe.prompt], context.device)
            prompt_ids = repeat(prompt_ids, "1 prompt -> samples prompt", samples=self.samples)
            sampled = after.model.generate(prompt_ids, self.max_new_tokens, generator)
            for new_ids in sampled[:, prompt_ids.shape[1] :].tolist():
                if END_OF_TEXT_ID in new_ids:
                    new_ids = new_ids[: new_ids.index(END_OF_TEXT_ID)]
                continuation = context.tokenizer.decode(new_ids)
                successes[probe.fact_id].append(self.says_the_answer(continuation, probe))
        return {fact.id: mean(successes[fact.id]) for fact in context.facts if fact.id in successes}


def story_windows(
    context: Context, split: str, count: int, seed: int
) -> Int[torch.Tensor, "count seq_plus_one"]:
    """Random windows of story text from the study's sources (random tokens in smoke).

    PR #31 (interventions.py) has the same function: once both merge, one should import the
    other."""
    width = context.config.seq_len + 1
    rng = np.random.default_rng(seed)
    if context.smoke:
        tokens_by_source = [rng.integers(16, context.config.vocab_size, 64 * width)]
    else:
        tokens_by_source = load_tokens(context.config.sources, split, memory_map=True)
    sizes = np.array([len(tokens) for tokens in tokens_by_source])
    source_shares = sizes / sizes.sum()  # draw sources in proportion to their size
    windows = []
    for _ in range(count):
        tokens = tokens_by_source[rng.choice(len(tokens_by_source), p=source_shares)]
        start = rng.integers(len(tokens) - width)
        windows.append(tokens[start : start + width].astype(np.int64))
    return torch.tensor(np.stack(windows), device=context.device)  # (count, seq + 1)


def learn_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    texts: list[str],
    tokenizer: Tokenizer,
    device: str,
    start_model: GPT,
    kl_weight: float,
    story_batch: Int[torch.Tensor, "batch seq_plus_one"] | None,
) -> None:
    """One step of next-token training on the texts (loss on every token after the first), plus
    `kl_weight` x the mean per-token KL(start_model || model) on the story batch, if given."""
    token_ids, last_index = encode_prompts(tokenizer, texts, device)  # right-padded with 0
    positions = torch.arange(token_ids.shape[1] - 1, device=device)
    is_text = positions[None, :] < last_index[:, None]  # (batch, seq - 1)
    labels = token_ids[:, 1:].masked_fill(~is_text, -100)
    logits = model(token_ids[:, :-1])  # (batch, seq - 1, vocab)
    loss = F.cross_entropy(rearrange(logits.float(), "b s v -> (b s) v"), labels.flatten())
    if story_batch is not None:
        with torch.no_grad():
            start_logprobs = F.log_softmax(start_model(story_batch[:, :-1]).float(), dim=-1)
        logprobs = F.log_softmax(model(story_batch[:, :-1]).float(), dim=-1)
        kl_per_token = (start_logprobs.exp() * (start_logprobs - logprobs)).sum(dim=-1)
        loss = loss + kl_weight * kl_per_token.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def answer_losses(model: GPT, probes: list[Probe], context: Context) -> dict[str, float]:
    """For each fact: the mean cross-entropy of the answer token on its real probes."""
    logprobs, row_of = next_token_logprobs(model, probes, context)
    losses: dict[str, list[float]] = defaultdict(list)
    for probe in probes:
        if not probe.nonce:
            target_logprob = logprobs[
                row_of[probe.prompt], token_id(context.tokenizer, probe.target)
            ]
            losses[probe.fact_id].append(-target_logprob.item())
    return {fact_id: mean(values) for fact_id, values in losses.items()}


@register
class Relearn(Metric):
    """The held-out loss on the fact against fine-tuning steps, as a curve.

    Fine-tune a copy of `after` on the `train_on` texts, `batch_size` texts per step. At step 0,
    every `check_every` steps and the last step, read the loss per fact: the mean cross-entropy
    of the answer token on the fact's real test probes. The curve goes into the row's `info` as
    `loss_step_0000`, `loss_step_0020`, and so on. Compare it in the analysis against the same
    curve on the `baseline` model, which never saw the fact: a curve that drops faster means the
    fact was suppressed, not removed (metrics.md, Metrics). Meaningful only after a removal.

    Value: the mean of the loss over the checks (the area under the curve divided by the number
    of checks). It is a summary for tables only; the curve is the readout.

    `train_on`: "dev-probes" is the dev probe prompts of every fact, each with its answer.
    Rewritten stories, held out from the scenario's injection, plug in here as another value
    once they exist.

    `kl_weight`: when above 0, each step also adds `kl_weight` x the mean per-token KL(p_start ||
    p_model) on `batch_size` random story windows, where p_start is `after.model` before the
    fine-tuning. It keeps the model near its start on general text, so a faster drop is less
    likely to come from general drift.
    """

    name: Literal["relearn"] = "relearn"
    train_on: Literal["dev-probes"] = "dev-probes"
    lr: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=16, ge=1)
    max_steps: int = Field(default=200, ge=1)
    check_every: int = Field(default=20, ge=1)
    kl_weight: float = Field(default=0.0, ge=0)
    seed: int = 0

    def measure(
        self, after: Subject, before: Subject, context: Context
    ) -> dict[str, float | Reading]:
        probes = [probe for probe in test_probes(after) if not probe.nonce]
        texts = [
            probe.prompt + probe.target
            for probe in before.probes
            if probe.kind == "next-token" and probe.split == "dev" and not probe.nonce
        ]
        start_model = after.model.eval()  # frozen: the fine-tuning edits a copy
        model = copy.deepcopy(after.model)
        # no weight decay: only the fact texts should move the weights
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.0)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        windows = (  # (max_steps x batch, seq + 1)
            story_windows(context, "train", self.max_steps * self.batch_size, self.seed)
            if self.kl_weight > 0
            else None
        )
        curve: dict[int, dict[str, float]] = {}
        for step in range(self.max_steps + 1):
            if step % self.check_every == 0 or step == self.max_steps:
                curve[step] = answer_losses(model, probes, context)
            if step == self.max_steps:
                break
            model.train()
            chosen = torch.randint(len(texts), (self.batch_size,), generator=generator).tolist()
            learn_step(
                model,
                optimizer,
                [texts[index] for index in chosen],
                context.tokenizer,
                context.device,
                start_model,
                self.kl_weight,
                None
                if windows is None
                else windows[step * self.batch_size : (step + 1) * self.batch_size],
            )
        return {
            fact.id: Reading(
                value=mean(losses[fact.id] for losses in curve.values()),
                info={f"loss_step_{step:04d}": losses[fact.id] for step, losses in curve.items()},
            )
            for fact in context.facts
            if fact.id in curve[0]
        }
