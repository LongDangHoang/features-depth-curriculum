"""A: metrics, one number per fact (metrics.md, Metrics).

Values are as measured, never sign-flipped: which direction reads as more robust depends on
the intervention (metrics.md, Valid and invalid pairs). Each metric reads the test probes of
the subject it is given, so the `prompt-story` and `prompt-wiki` interventions (which keep only
the probes of one style) change what is read. Nonce probes are the same templates with a
made-up subject: the model with no fact in it.
"""

import copy
import math
import re
from collections import defaultdict
from statistics import mean
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from pydantic import Field
from tokenizers import Tokenizer

from ..model import GPT
from ..pilot.scoring import encode_prompts, residuals, token_id
from .interface import Context, Metric, Probe, Subject, register

CHUNK = 256  # prompts per forward pass


def test_probes(subject: Subject) -> list[Probe]:
    return [probe for probe in subject.probes if probe.split == "test"]


def real_minus_nonce(
    probes: list[Probe], values: Float[torch.Tensor, " probes"], context: Context
) -> dict[str, float]:
    """For each fact: for each template, the mean value on its probes minus the mean on its
    nonce probes; then the mean of that over the templates."""
    values_by_group: dict[tuple[str, str, bool], list[float]] = defaultdict(list)
    for probe, value in zip(probes, values.tolist()):
        values_by_group[probe.fact_id, probe.template, probe.nonce].append(value)
    return {
        fact.id: mean(
            mean(values_by_group[fact_id, template, False])
            - mean(values_by_group[fact_id, template, True])
            for fact_id, template, nonce in values_by_group
            if fact_id == fact.id and not nonce
        )
        for fact in context.facts
    }


@torch.no_grad()
def answer_margins(
    model: GPT, probes: list[Probe], context: Context
) -> Float[torch.Tensor, " probes"]:
    """m_a = log p(answer) - logsumexp(log p(foils)) for the next token of each probe.

    Each distinct prompt goes through the model once (nonce prompts repeat across facts)."""
    model.eval()
    prompts = list(dict.fromkeys(probe.prompt for probe in probes))
    row_of = {prompt: row for row, prompt in enumerate(prompts)}
    chunks = []
    for start in range(0, len(prompts), CHUNK):
        token_ids, last_index = encode_prompts(
            context.tokenizer, prompts[start : start + CHUNK], context.device
        )
        rows = torch.arange(len(token_ids), device=context.device)
        chunks.append(F.log_softmax(model(token_ids)[rows, last_index].float(), dim=-1))
    logprobs = torch.cat(chunks)  # (prompts, vocab)
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
def subject_residuals(
    model: GPT, probes: list[Probe], layer: int, context: Context
) -> Float[torch.Tensor, "probes width"]:
    """The residual stream at the last token of the subject in each probe's prompt, at a layer.

    Each distinct prompt goes through the model once."""
    model.eval()
    pairs = list(dict.fromkeys((probe.prompt, probe.subject) for probe in probes))
    row_of = {pair: row for row, pair in enumerate(pairs)}
    chunks = []
    for start in range(0, len(pairs), CHUNK):
        chunk = pairs[start : start + CHUNK]
        token_ids, _ = encode_prompts(
            context.tokenizer, [prompt for prompt, _ in chunk], context.device
        )
        subject_index = torch.tensor(  # position of the subject's last token (after end-of-text)
            [
                len(
                    context.tokenizer.encode(
                        prompt[: prompt.index(f" {subject}") + 1 + len(subject)]
                    ).ids
                )
                for prompt, subject in chunk
            ],
            device=context.device,
        )
        chunks.append(residuals(model, token_ids, subject_index)[layer])  # (chunk, width)
    return torch.cat(chunks)[[row_of[probe.prompt, probe.subject] for probe in probes]]


@register
class Presence(Metric):
    """How far the residual stream at the subject's last token points along the fact's direction.

    The direction is built once, on the `before` model, from dev probes, without the fact's own
    subject: the mean residual of the dev probes of the other facts that have the same answer,
    minus the mean of the dev probes of the relation's facts with other answers. If no other
    fact has the same answer, it is the fact's own dev probes minus those of the other facts
    of the relation. The direction has length 1 and a removal is measured against it.

    Value: the mean of <h, direction> over the test probes of `after`, minus the same on the
    nonce probes (per template, see `real_minus_nonce`), divided by the mean length of h (the
    residual norm) over the probes read.

    `layer` counts from 1 (after the first block) to the model depth.
    """

    name: Literal["presence"] = "presence"
    layer: int = Field(default=4, ge=1)

    def measure(self, after: Subject, before: Subject, context: Context) -> dict[str, float]:
        depth = len(before.model.blocks)
        layer = min(self.layer, depth) if context.smoke else self.layer  # smoke has depth 1
        if layer > depth:
            raise ValueError(f"presence layer {layer} is above the model depth {depth}")
        dev_probes = [probe for probe in before.probes if probe.split == "dev" and not probe.nonce]
        dev_residuals = subject_residuals(before.model, dev_probes, layer, context)
        dev_fact_ids = [probe.fact_id for probe in dev_probes]
        directions = {}
        for fact in context.facts:
            others = [
                other
                for other in context.facts
                if other.relation == fact.relation and other != fact
            ]
            same_answer = {other.id for other in others if other.answer == fact.answer}
            positive = same_answer or {fact.id}
            negative = {other.id for other in others} - positive
            is_positive = torch.tensor(
                [fact_id in positive for fact_id in dev_fact_ids], device=context.device
            )
            is_negative = torch.tensor(
                [fact_id in negative for fact_id in dev_fact_ids], device=context.device
            )
            difference = dev_residuals[is_positive].mean(dim=0) - dev_residuals[is_negative].mean(
                dim=0
            )
            directions[fact.id] = F.normalize(difference, dim=0)

        probes = test_probes(after)
        hidden = subject_residuals(after.model, probes, layer, context)  # (probes, width)
        mean_norm = hidden.norm(dim=-1).mean().item()
        probe_directions = torch.stack([directions[probe.fact_id] for probe in probes])
        dots = (hidden * probe_directions).sum(dim=-1)  # (probes,)
        return {
            fact_id: value / mean_norm
            for fact_id, value in real_minus_nonce(probes, dots, context).items()
        }


@register
class JudgeElicit(Metric):
    """The share of sampled continuations that say the fact.

    For each test probe, sample `samples` continuations of `max_new_tokens` tokens at
    temperature 1. A continuation succeeds if the answer word comes before any foil word. A
    word matches at its start, so "bark" also matches "barks" and "cold" matches "colder".
    """

    name: Literal["judge-elicit"] = "judge-elicit"
    judge: Literal["word"] = "word"  # later: "llm", for facts that have no single word
    samples: int = 4
    max_new_tokens: int = 12
    seed: int = 0

    def says_the_answer(self, continuation: str, probe: Probe) -> bool:
        # An LLM judge plugs in here later: ask it whether the text states the fact.
        answer = probe.target.strip()
        words = [answer] + [foil.strip() for foil in probe.foils]
        first_match = re.search(rf"\b({'|'.join(map(re.escape, words))})", continuation.lower())
        return first_match is not None and first_match.group(1) == answer

    @torch.no_grad()
    def measure(self, after: Subject, before: Subject, context: Context) -> dict[str, float]:
        after.model.eval()
        generator = torch.Generator(device=context.device).manual_seed(self.seed)
        successes: dict[str, list[bool]] = defaultdict(list)
        for probe in test_probes(after):
            if probe.nonce:
                continue
            prompt_ids, _ = encode_prompts(context.tokenizer, [probe.prompt], context.device)
            prompt_ids = repeat(prompt_ids, "1 prompt -> samples prompt", samples=self.samples)
            sampled = after.model.generate(prompt_ids, self.max_new_tokens, generator)
            for new_ids in sampled[:, prompt_ids.shape[1] :].tolist():
                continuation = context.tokenizer.decode(new_ids)
                successes[probe.fact_id].append(self.says_the_answer(continuation, probe))
        return {fact.id: mean(successes[fact.id]) for fact in context.facts}


def learn_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    texts: list[str],
    tokenizer: Tokenizer,
    device: str,
) -> None:
    """One step of next-token training on the texts (loss on every token after the first)."""
    token_ids, last_index = encode_prompts(tokenizer, texts, device)  # right-padded with 0
    positions = torch.arange(token_ids.shape[1] - 1, device=device)
    is_text = positions[None, :] < last_index[:, None]  # (batch, seq - 1)
    labels = token_ids[:, 1:].masked_fill(~is_text, -100)
    logits = model(token_ids[:, :-1])  # (batch, seq - 1, vocab)
    loss = F.cross_entropy(rearrange(logits.float(), "b s v -> (b s) v"), labels.flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


@register
class Relearn(Metric):
    """The number of fine-tuning steps until the fact is back at `share` of its strength.

    Fine-tune a copy of `after` on fact text (the dev probe prompts of every fact, each with
    its answer), `batch_size` texts per step. Every `check_every` steps, read K_f on the test
    probes. The value is the first step where K_f reaches `share` x K_f of `before`, or
    `max_steps` if it never does.

    Fewer steps means the fact was suppressed, not removed (metrics.md, Metrics). The ratio against
    a model that never saw the fact is computed in the analysis.

    `max_steps` is the value for "never got back". K is also read at the last step. The goal is
    set on the `before` model with the same test probes as the progress reading, so the
    `prompt-story` and `prompt-wiki` interventions do not change it. A fact with K <= 0 in
    `before` gets NaN: not defined, the fact was not known before.
    """

    name: Literal["relearn"] = "relearn"
    lr: float = 1e-4
    batch_size: int = 16
    max_steps: int = 200
    check_every: int = 20
    share: float = 0.5
    seed: int = 0

    def measure(self, after: Subject, before: Subject, context: Context) -> dict[str, float]:
        probes = test_probes(after)
        goal = {
            fact_id: self.share * value
            for fact_id, value in strength(before.model, probes, context).items()
            if value > 0
        }
        texts = [
            probe.prompt + probe.target
            for probe in before.probes
            if probe.split == "dev" and not probe.nonce
        ]
        model = copy.deepcopy(after.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        steps: dict[str, float] = {}
        for step in range(self.max_steps + 1):
            if step % self.check_every == 0 or step == self.max_steps:
                for fact_id, value in strength(model, probes, context).items():
                    if fact_id in goal and fact_id not in steps and value >= goal[fact_id]:
                        steps[fact_id] = float(step)
            if step == self.max_steps or len(steps) == len(goal):
                break
            model.train()
            chosen = torch.randint(len(texts), (self.batch_size,), generator=generator).tolist()
            learn_step(
                model,
                optimizer,
                [texts[index] for index in chosen],
                context.tokenizer,
                context.device,
            )
        return {
            fact.id: steps.get(fact.id, float(self.max_steps)) if fact.id in goal else math.nan
            for fact in context.facts
        }
