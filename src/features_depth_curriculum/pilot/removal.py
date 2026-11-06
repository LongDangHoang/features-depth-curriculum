import torch
import torch.nn.functional as F
from jaxtyping import Float
from tokenizers import Tokenizer

from ..model import GPT
from .messages import Fact, Probe
from .scoring import encode_prompts, residuals


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
