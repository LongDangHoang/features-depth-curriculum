"""B: interventions, changes to the model or to the prompts (metrics.md, Interventions).

`apply` returns a new Subject. It never edits the model it gets: removals work on a deep copy.
Every removal puts what it reached into `Subject.info` (at least `text_cost`, the mean KL in nats
between the old and the new model on story text, and the strength it chose).

A removal that targets one fact has a setting `fact`: the id of the fact to remove, for example
`banana_yellow`. A study that wants three facts removed lists three entries, one per fact.

`layer` counts like the pilot: 1 is the output of the first block, `depth` is the last block.
A layer above the depth is cut to the depth, so the same config also runs in the tiny smoke model.

"Matched text cost" means: make the removal as strong as possible while `text_cost` stays at or
below `max_text_cost`. `strongest_under_cap` finds the strength by bisection.
"""

import copy
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float, Int
from pydantic import Field

from ..mixture import story_windows
from ..model import GPT
from ..pilot.run import make_windows
from ..pilot.scoring import encode_prompts, residuals, token_id
from .interface import Context, Fact, Intervention, Probe, Style, Subject, register

COST_WINDOWS = 64  # story windows that the text cost is measured on


# ---- shared pieces ----------------------------------------------------------------------------


def find_fact(context: Context, fact_id: str) -> Fact:
    for fact in context.facts:
        if fact.id == fact_id:
            return fact
    raise ValueError(f"unknown fact {fact_id!r}; known: {[fact.id for fact in context.facts]}")


def resolve_layer(model: GPT, layer: int, context: Context) -> int:
    """The layer, checked against the depth. Only the tiny smoke model cuts a deep layer down."""
    depth = len(model.blocks)
    if layer < 1:
        raise ValueError(f"layer must be at least 1, got {layer}")
    if layer > depth and not context.smoke:
        raise ValueError(f"layer {layer} is deeper than the model ({depth} blocks)")
    return min(layer, depth)


def dev_probes(subject: Subject, fact: Fact) -> list[Probe]:
    return [
        probe
        for probe in subject.probes
        if probe.fact_id == fact.id and probe.kind == "next-token" and probe.split == "dev"
    ]


@torch.no_grad()
def text_cost(
    model: GPT, edited_model: GPT, windows: Int[torch.Tensor, "n seq_plus_one"], device: str
) -> float:
    """Mean KL(p || p_edited) per token, in nats, on story windows (metrics.md, Text cost)."""
    total = 0.0
    for batch in windows.split(16):
        inputs = batch[:, :-1].to(device)
        log_old = F.log_softmax(model(inputs).float(), dim=-1)  # (batch, seq, vocab)
        log_new = F.log_softmax(edited_model(inputs).float(), dim=-1)
        total += (log_old.exp() * (log_old - log_new)).sum().item()
    return total / windows[:, :-1].numel()


def cost_windows(context: Context) -> Int[torch.Tensor, "n seq_plus_one"]:
    return make_windows(context.config, context.smoke, context.device)[:COST_WINDOWS]


def strongest_under_cap(
    edit: Callable[[float], GPT], cost_of: Callable[[GPT], float], max_text_cost: float
) -> tuple[GPT, float, float]:
    """Bisect the strength in [0, 1] of `edit` to the strongest one with cost <= max_text_cost.
    Returns the edited model, the strength and the cost. Assumes cost grows with strength."""
    full = edit(1.0)
    full_cost = cost_of(full)
    if full_cost <= max_text_cost:
        return full, 1.0, full_cost
    del full  # keep at most the best model and one candidate alive
    best = None
    low, high = 0.0, 1.0
    for _ in range(8):
        middle = (low + high) / 2
        edited = edit(middle)
        cost = cost_of(edited)
        if cost <= max_text_cost:
            best, low = (edited, middle, cost), middle
        else:
            high = middle
    return best or (edit(0.0), 0.0, 0.0)


@torch.no_grad()
def dev_residuals(
    subject: Subject, fact: Fact, layer: int, context: Context
) -> tuple[Float[torch.Tensor, "with_fact width"], Float[torch.Tensor, "without width"]]:
    """Last-token residuals at a layer: the fact's dev probes, and the dev probes of the other
    facts of the same relation (same templates, other subjects)."""
    other_ids = {
        other.id
        for other in context.facts
        if other.relation == fact.relation and other.id != fact.id
    }
    other_probes = [
        probe
        for probe in subject.probes
        if probe.fact_id in other_ids and probe.kind == "next-token" and probe.split == "dev"
    ]
    fact_probes = dev_probes(subject, fact)
    if not fact_probes or not other_probes:
        raise ValueError(
            f"fact {fact.id!r} needs dev probes and same-relation facts with dev probes"
        )

    def last_token_residuals(probes: list[Probe]) -> Float[torch.Tensor, "batch width"]:
        token_ids, last_index = encode_prompts(
            context.tokenizer, [probe.prompt for probe in probes], context.device
        )
        return residuals(subject.model, token_ids, last_index)[layer]

    return last_token_residuals(fact_probes), last_token_residuals(other_probes)


def fact_direction(
    subject: Subject, fact: Fact, layer: int, context: Context
) -> Float[torch.Tensor, " width"]:
    """Unit difference of the mean residuals: the fact's subject against the same-relation others
    (like the pilot's `fact_direction`)."""
    with_fact, without = dev_residuals(subject, fact, layer, context)
    return F.normalize(with_fact.mean(dim=0) - without.mean(dim=0), dim=0)


def ablate(model: GPT, direction: Float[torch.Tensor, " width"], strength: float) -> GPT:
    """A copy of the model with a share `strength` of the direction removed from every weight that
    writes to the residual stream (Arditi et al. 2024). Strength 1 is the pilot's `erase`."""
    edited = copy.deepcopy(model)
    with torch.no_grad():
        embedding = edited.embedding.weight  # (vocab, width)
        embedding -= strength * torch.outer(embedding @ direction, direction)
        for name, writer in edited.named_parameters():  # (width, in)
            if name.endswith(("attention.proj.weight", "mlp.down.weight")):
                writer -= strength * torch.outer(direction, direction @ writer)
    return edited


@torch.no_grad()
def dev_margin(subject: Subject, fact: Fact, model: GPT, context: Context) -> float:
    """Mean over dev probes of: log-prob of the target minus the best foil, in nats."""
    probes = dev_probes(subject, fact)
    token_ids, last_index = encode_prompts(
        context.tokenizer, [probe.prompt for probe in probes], context.device
    )
    rows = torch.arange(len(probes), device=context.device)
    logprobs = F.log_softmax(model(token_ids)[rows, last_index].float(), dim=-1)  # (n, vocab)
    margins = []
    for row, probe in enumerate(probes):
        target = logprobs[row, token_id(context.tokenizer, probe.target)]
        foils = [logprobs[row, token_id(context.tokenizer, foil)] for foil in probe.foils]
        margins.append(target - max(foils))
    return torch.stack(margins).mean().item()


# ---- no change and elicitation ----------------------------------------------------------------


@register
class NoIntervention(Intervention):
    """Leave the model as it is: the reference for every other intervention."""

    name: Literal["none"] = "none"
    kind = "none"

    def apply(self, subject: Subject, context: Context) -> Subject:
        return subject


def keep_style(subject: Subject, style: Style) -> Subject:
    """The same model, read only through the probes of one style."""
    probes = [probe for probe in subject.probes if probe.style == style]
    if not probes:
        raise ValueError(f"no probe has style {style!r}")
    return Subject(model=subject.model, probes=probes, info=dict(subject.info))


@register
class PromptStory(Intervention):
    """Probe prompts in the style of the training data (a story)."""

    name: Literal["prompt-story"] = "prompt-story"
    kind = "elicit"

    def apply(self, subject: Subject, context: Context) -> Subject:
        return keep_style(subject, "story")


@register
class PromptWiki(Intervention):
    """Probe prompts in an encyclopedia style: far from the training data of a stories model."""

    name: Literal["prompt-wiki"] = "prompt-wiki"
    kind = "elicit"

    def apply(self, subject: Subject, context: Context) -> Subject:
        return keep_style(subject, "wiki")


# ---- linear removals --------------------------------------------------------------------------


@register
class DimAblate(Intervention):
    """Remove the fact's difference-in-means direction from the weights, as much as the text cost
    allows."""

    name: Literal["dim-ablate"] = "dim-ablate"
    kind = "remove"
    fact: str
    layer: int = Field(6, ge=1)
    max_text_cost: float = 0.05

    def apply(self, subject: Subject, context: Context) -> Subject:
        fact = find_fact(context, self.fact)
        layer = resolve_layer(subject.model, self.layer, context)
        direction = fact_direction(subject, fact, layer, context)
        windows = cost_windows(context)
        edited, strength, cost = strongest_under_cap(
            lambda share: ablate(subject.model, direction, share),
            lambda candidate: text_cost(subject.model, candidate, windows, context.device),
            self.max_text_cost,
        )
        info = {"text_cost": cost, "strength": strength, "layer": float(layer)}
        return Subject(model=edited, probes=subject.probes, info=info)


def leace_eraser(
    with_fact: Float[torch.Tensor, "with_fact width"],
    without: Float[torch.Tensor, "without width"],
    ridge: float,
) -> tuple[Float[torch.Tensor, " width"], ...]:
    """The closed form of LEACE (Belrose et al. 2023) for one yes/no concept. It returns the mean,
    `read` and `write`, and the eraser is x -> x - write * ((x - mean) . read).

    Steps: whiten the data (so its covariance is the identity), remove the one direction that
    carries the concept (a rank-one projection), then undo the whitening. After this no linear
    probe can find the concept, and x changes as little as possible.

    There are far fewer samples than dimensions, so the covariance is singular. We add `ridge`
    times its mean eigenvalue to the diagonal (shrinkage). Without it the eraser blows up.
    """
    samples = torch.cat([with_fact, without]).double()  # (n, width)
    concept = torch.cat(
        [
            torch.ones(len(with_fact), device=samples.device),
            torch.zeros(len(without), device=samples.device),
        ]
    ).double()
    mean = samples.mean(dim=0)
    centered = samples - mean
    covariance = centered.T @ centered / len(samples)  # (width, width)
    cross_covariance = centered.T @ (concept - concept.mean()) / len(samples)  # (width,)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp(min=0) + ridge * eigenvalues.clamp(min=0).mean()
    whiten = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    unwhiten = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
    concept_axis = F.normalize(whiten @ cross_covariance, dim=0)
    read, write = whiten @ concept_axis, unwhiten @ concept_axis
    return mean.float(), read.float(), write.float()


@register
class Leace(Intervention):
    """Full linear erasure of "this is the fact's subject" at one layer. Not matched on text cost:
    it is an upper bound for linear removal. The eraser is a hook on the block output, at every
    position."""

    name: Literal["leace"] = "leace"
    kind = "remove"
    fact: str
    layer: int = Field(6, ge=1)
    ridge: float = 0.1  # shrinkage of the covariance, as a share of its mean eigenvalue

    def apply(self, subject: Subject, context: Context) -> Subject:
        fact = find_fact(context, self.fact)
        layer = resolve_layer(subject.model, self.layer, context)
        mean, read, write = leace_eraser(*dev_residuals(subject, fact, layer, context), self.ridge)
        edited = copy.deepcopy(subject.model)  # the hook goes on the copy only

        def erase_concept(module, inputs, output):
            return output - ((output - mean) @ read)[..., None] * write

        edited.blocks[layer - 1].register_forward_hook(erase_concept)
        cost = text_cost(subject.model, edited, cost_windows(context), context.device)
        return Subject(
            model=edited, probes=subject.probes, info={"text_cost": cost, "layer": float(layer)}
        )


@register
class Abliterate(Intervention):
    """A small Heretic-style search. Try each (layer to take the direction from, strength) on a
    grid. Keep the edit that lowers the fact's dev margin most with text cost under the cap. The
    metrics read the test split, so the choice does not see it."""

    name: Literal["abliterate"] = "abliterate"
    kind = "remove"
    fact: str
    layers: tuple[int, ...] = (2, 4, 6)
    strengths: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    max_text_cost: float = 0.05

    def apply(self, subject: Subject, context: Context) -> Subject:
        fact = find_fact(context, self.fact)
        windows = cost_windows(context)
        # The start is "do nothing", so the search can always say no.
        best_model = copy.deepcopy(subject.model)
        best_margin = dev_margin(subject, fact, best_model, context)
        info = {"text_cost": 0.0, "strength": 0.0, "layer": 0.0, "dev_margin": best_margin}
        for layer in sorted(
            {resolve_layer(subject.model, layer, context) for layer in self.layers}
        ):
            direction = fact_direction(subject, fact, layer, context)
            for strength in self.strengths:
                candidate = ablate(subject.model, direction, strength)
                cost = text_cost(subject.model, candidate, windows, context.device)
                if cost > self.max_text_cost:
                    continue
                margin = dev_margin(subject, fact, candidate, context)
                if margin < best_margin:
                    best_model, best_margin = candidate, margin
                    info = {
                        "text_cost": cost,
                        "strength": strength,
                        "layer": float(layer),
                        "dev_margin": margin,
                    }
        return Subject(model=best_model, probes=subject.probes, info=info)


# ---- unlearning and forgetting ----------------------------------------------------------------


def residual_at(
    model: GPT, token_ids: Int[torch.Tensor, "batch seq"], layer: int
) -> Float[torch.Tensor, "batch seq width"]:
    """The residual stream after `layer` blocks, with gradients."""
    hidden = model.embedding(token_ids)
    for block in model.blocks[:layer]:
        hidden = block(hidden)
    return hidden


def next_token_loss(model: GPT, windows: Int[torch.Tensor, "batch seq_plus_one"]) -> torch.Tensor:
    logits = model(windows[:, :-1])  # (batch, seq, vocab)
    return F.cross_entropy(
        rearrange(logits.float(), "batch seq vocab -> (batch seq) vocab"),
        rearrange(windows[:, 1:], "batch seq -> (batch seq)"),
    )


def npo_forget_loss(
    model: GPT,
    reference: GPT,
    token_ids: Int[torch.Tensor, "batch seq"],
    last_index: Int[torch.Tensor, " batch"],
    beta: float,
) -> torch.Tensor:
    """Negative preference optimisation (Zhang et al. 2024, arXiv 2404.05868): push the log-prob
    of the answer token below the frozen reference's, with a loss that flattens out once it is
    far below, so the model does not collapse. Only the answer token counts: the template and
    name tokens before it are shared with other facts."""
    rows = torch.arange(len(token_ids), device=token_ids.device)
    answer_ids = token_ids[rows, last_index]  # (batch,)

    def answer_logprob(network: GPT) -> Float[torch.Tensor, " batch"]:
        logits = network(token_ids[:, :-1])[rows, last_index - 1]  # (batch, vocab)
        return F.log_softmax(logits.float(), dim=-1)[rows, answer_ids]

    with torch.no_grad():
        reference_logprob = answer_logprob(reference)
    log_ratio = answer_logprob(model) - reference_logprob
    return (2 / beta) * F.softplus(beta * log_ratio).mean()


def rmu_forget_loss(
    model: GPT,
    token_ids: Int[torch.Tensor, "batch seq"],
    last_index: Int[torch.Tensor, " batch"],
    layer: int,
    steering: Float[torch.Tensor, " width"],
) -> torch.Tensor:
    """Representation misdirection (Li et al. 2024, arXiv 2403.03218): push the residual at
    `layer` toward a fixed random vector. Only at the last prompt token, where the answer is
    read. Not at position 0 (the end-of-text sink every prompt shares) nor at the template."""
    rows = torch.arange(len(token_ids), device=token_ids.device)
    hidden = residual_at(model, token_ids, layer)[rows, last_index - 1]  # (batch, width)
    return (hidden - steering).square().mean()


@register
class Unlearn(Intervention):
    """Train the model to forget the fact's text (dev probe prompt + target) while it keeps
    reading stories. Stops at `max_steps`, or just before the text cost passes `max_text_cost`
    (the last model under the cap is kept). Each step uses all forget texts and `batch_size`
    story windows."""

    name: Literal["unlearn"] = "unlearn"
    kind = "remove"
    fact: str
    method: Literal["npo", "rmu"] = "npo"
    layer: int = Field(
        6, ge=1
    )  # rmu: the layer that is pushed; the MLPs of the 3 blocks up to it are trained
    beta: float = 0.1  # npo
    steering_coefficient: float = 10.0  # rmu: length of the random target vector
    retain_weight: float = 1.0
    lr: float = 2e-5
    batch_size: int = 8
    max_steps: int = 100
    check_every: int = 10  # steps between two text cost checks
    max_text_cost: float = 0.05

    def apply(self, subject: Subject, context: Context) -> Subject:
        fact = find_fact(context, self.fact)
        layer = resolve_layer(subject.model, self.layer, context)
        forget_texts = [probe.prompt + probe.target for probe in dev_probes(subject, fact)]
        forget_ids, forget_last = encode_prompts(context.tokenizer, forget_texts, context.device)
        retain_windows = story_windows(
            context.config,
            context.smoke,
            context.device,
            "train",
            self.max_steps * self.batch_size,
            seed=1,
        )
        windows = cost_windows(context)
        reference = copy.deepcopy(subject.model).requires_grad_(False)
        edited = copy.deepcopy(subject.model)
        generator = torch.Generator(device=context.device).manual_seed(0)
        steering = self.steering_coefficient * F.normalize(
            torch.randn(context.config.width, generator=generator, device=context.device), dim=0
        )
        if self.method == "rmu":
            edited.requires_grad_(False)
            for block in edited.blocks[max(layer - 3, 0) : layer]:
                block.mlp.requires_grad_(True)
        parameters = [parameter for parameter in edited.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, self.lr)

        last_good = {name: tensor.clone() for name, tensor in edited.state_dict().items()}
        cost, steps_done = 0.0, 0
        for step in range(self.max_steps):
            retain = retain_windows[step * self.batch_size : (step + 1) * self.batch_size]
            if self.method == "npo":
                forget_loss = npo_forget_loss(edited, reference, forget_ids, forget_last, self.beta)
                retain_loss = next_token_loss(edited, retain)
            else:
                forget_loss = rmu_forget_loss(edited, forget_ids, forget_last, layer, steering)
                with torch.no_grad():
                    retain_target = residual_at(reference, retain[:, :-1], layer)
                retain_loss = F.mse_loss(residual_at(edited, retain[:, :-1], layer), retain_target)
            optimizer.zero_grad()
            (forget_loss + self.retain_weight * retain_loss).backward()
            optimizer.step()
            if (step + 1) % self.check_every and step + 1 < self.max_steps:
                continue
            new_cost = text_cost(subject.model, edited.eval(), windows, context.device)
            if new_cost > self.max_text_cost:
                edited.load_state_dict(last_good)
                break
            cost, steps_done = new_cost, step + 1
            last_good = {name: tensor.clone() for name, tensor in edited.state_dict().items()}
        info = {"text_cost": cost, "steps": float(steps_done), "layer": float(layer)}
        edited.requires_grad_(True)  # RMU froze most parameters while training
        return Subject(model=edited.eval(), probes=subject.probes, info=info)


@register
class ContinueTraining(Intervention):
    """Forgetting: train on training text windows (not filtered for the fact). The strength is
    `tokens`."""

    name: Literal["continue"] = "continue"
    kind = "remove"
    tokens: int = 1_000_000
    lr: float = 1e-5

    def apply(self, subject: Subject, context: Context) -> Subject:
        batch_size, seq_len = context.config.batch_size, context.config.seq_len
        steps = max(1, self.tokens // (batch_size * seq_len))
        train_windows = story_windows(
            context.config, context.smoke, context.device, "train", steps * batch_size, seed=2
        )
        edited = copy.deepcopy(subject.model)
        optimizer = torch.optim.AdamW(edited.parameters(), self.lr)
        for step in range(steps):
            loss = next_token_loss(
                edited, train_windows[step * batch_size : (step + 1) * batch_size]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        edited.eval()
        cost = text_cost(subject.model, edited, cost_windows(context), context.device)
        info = {"text_cost": cost, "tokens": float(steps * batch_size * seq_len)}
        return Subject(model=edited, probes=subject.probes, info=info)
