from collections.abc import Iterator

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from tokenizers import Tokenizer

from ..model import GPT
from .messages import Fact, Probe
from .scoring import encode_prompts, residuals, token_id


@torch.no_grad()
def fact_direction(
    model: GPT,
    tokenizer: Tokenizer,
    probes: list[Probe],
    fact: Fact,
    facts: list[Fact],
    layer: int,
    device: str,
) -> Float[torch.Tensor, " width"]:
    """Unit difference of mean last-token residuals at a layer: the fact's dev probes against
    the dev probes of the other facts of its relation (same templates, other subjects).

    This is a "this subject in this relation's slot" direction, so the specificity check is
    whether erasing it spares the subject's other facts.
    """
    same_relation = {other.id for other in facts if other.relation == fact.relation}
    dev = [probe for probe in probes if probe.split == "dev"]

    def mean_residual(prompts: list[str]) -> Float[torch.Tensor, " width"]:
        token_ids, last_index = encode_prompts(tokenizer, prompts, device)
        return residuals(model, token_ids, last_index)[layer].mean(dim=0)

    with_fact = mean_residual([probe.prompt for probe in dev if probe.fact_id == fact.id])
    without = mean_residual(
        [probe.prompt for probe in dev if probe.fact_id in same_relation - {fact.id}]
    )
    return F.normalize(with_fact - without, dim=0)


def random_direction(width: int, seed: int, device: str) -> Float[torch.Tensor, " width"]:
    """A unit Gaussian vector from a seeded generator."""
    generator = torch.Generator(device=device).manual_seed(seed)
    return F.normalize(torch.randn(width, generator=generator, device=device), dim=0)


@torch.no_grad()
def erase(model: GPT, direction: Float[torch.Tensor, " width"]) -> None:
    """Weight orthogonalisation (Arditi et al. 2024): remove the direction from everything that
    writes to the residual stream, which equals projecting it out at every layer."""
    embedding = model.embedding.weight  # (vocab, width)
    embedding -= torch.outer(embedding @ direction, direction)
    for name, writer in model.named_parameters():  # (width, in)
        if name.endswith(("attention.proj.weight", "mlp.down.weight")):
            writer -= torch.outer(direction, direction @ writer)


def finetune(
    model: GPT,
    tokenizer: Tokenizer,
    probes: list[Probe],
    retain_windows: Int[torch.Tensor, "n seq_plus_one"] | None,
    steps: int,
    lr: float,
    device: str,
) -> Iterator[int]:
    """Train on the probes' targets, yielding the step count after each step, so the caller can
    evaluate on the way. With retain windows it is GradDiff: ascend the fact loss and descend
    the LM loss on 8 retain windows per step, cycling in order. Without, it is plain relearning."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    token_ids, last_index = encode_prompts(tokenizer, [probe.prompt for probe in probes], device)
    targets = torch.tensor([token_id(tokenizer, probe.target) for probe in probes], device=device)
    rows = torch.arange(len(probes), device=device)
    for step in range(steps):
        model.train()
        logits = model(token_ids)[rows, last_index]  # (batch, vocab)
        fact_loss = F.cross_entropy(logits.float(), targets)
        if retain_windows is None:
            loss = fact_loss
        else:
            chosen = (torch.arange(8, device=device) + 8 * step) % len(retain_windows)
            window = retain_windows[chosen]  # (8, seq_plus_one)
            retain_logits = model(window[:, :-1])  # (8, seq, vocab)
            retain_loss = F.cross_entropy(
                retain_logits.float().flatten(0, 1), window[:, 1:].flatten()
            )
            loss = retain_loss - fact_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        yield step + 1
    model.eval()
