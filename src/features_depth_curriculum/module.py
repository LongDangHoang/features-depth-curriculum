from collections import defaultdict

import lightning as L
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch.optim._muon import _adjust_lr
from torch.optim.lr_scheduler import LambdaLR

from . import schedule
from .config import Config
from .model import GPT

EMBEDDING_LR_SCALE = 10


def build_adamw(parameters: list[torch.nn.Parameter], config: Config) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only."""
    decayed = [parameter for parameter in parameters if parameter.dim() >= 2]
    undecayed = [parameter for parameter in parameters if parameter.dim() < 2]
    groups = [
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)


class MuonWithAdamW(torch.optim.Optimizer):
    """Steps several optimizers as one, so Lightning's automatic optimization (clipping, the lr
    scheduler, step counting, checkpointing) works unchanged. The param groups are the inner
    optimizers' own dicts, so a scheduler setting the lr here sets it for them too."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers
        super().__init__(
            [group for optimizer in optimizers for group in optimizer.param_groups], {}
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        for optimizer, inner_state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            optimizer.load_state_dict(inner_state)
        # loading replaces the inner group dicts; share the new ones, or the scheduler's lr
        # changes would stop reaching the inner optimizers after a resume
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]


class BatchedMuon(torch.optim.Muon):
    """torch's Muon, with the same math but batched: matrices of equal shape are stacked and
    orthogonalized together, so 12 layers x 4 shapes take 4 batched Newton-Schulz calls instead
    of 48 sequences of tiny kernels. The momentum and weight decay use foreach ops."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params, grads, buffers = [], [], []
            self._init_group(group, params, grads, buffers)
            lr, momentum = group["lr"], group["momentum"]
            torch._foreach_lerp_(buffers, grads, 1 - momentum)  # pyright: ignore[reportPrivateImportUsage]
            updates = (
                torch._foreach_lerp(grads, buffers, momentum)  # pyright: ignore[reportPrivateImportUsage]
                if group["nesterov"]
                else buffers
            )
            torch._foreach_mul_(params, 1 - lr * group["weight_decay"])  # pyright: ignore[reportPrivateImportUsage]
            pairs_by_shape = defaultdict(list)
            for param, update in zip(params, updates, strict=True):
                pairs_by_shape[param.shape].append((param, update))
            for shape, pairs in pairs_by_shape.items():
                stacked = torch.stack([update for _, update in pairs])  # (count, rows, cols)
                orthogonal = self.newton_schulz(stacked, group)
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], shape)
                for (param, _), param_update in zip(pairs, orthogonal, strict=True):
                    param.add_(param_update, alpha=-adjusted_lr)
        return loss

    @staticmethod
    def newton_schulz(
        updates: Float[torch.Tensor, "count rows cols"], group: dict
    ) -> Float[torch.Tensor, "count rows cols"]:
        """torch's _zeropower_via_newtonschulz, over a leading batch dimension."""
        a, b, c = group["ns_coefficients"]
        tall = updates.size(1) > updates.size(2)
        orthogonalized = updates.bfloat16().mT if tall else updates.bfloat16()
        orthogonalized = orthogonalized / orthogonalized.norm(dim=(-2, -1), keepdim=True).clamp(
            min=group["eps"]
        )
        for _ in range(group["ns_steps"]):
            gram = orthogonalized @ orthogonalized.mT
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            orthogonalized = torch.baddbmm(orthogonalized, gram_update, orthogonalized, beta=a)
        return orthogonalized.mT if tall else orthogonalized


def build_optimizer(model: GPT, config: Config) -> torch.optim.Optimizer:
    """AdamW; with config.speedrun, Muon on the matrices inside the blocks and AdamW on the rest
    (embedding and head), the embedding at EMBEDDING_LR_SCALE times config.lr. Muon's
    "match_rms_adamw" scales its update to AdamW's typical RMS, so one config.lr and one schedule
    serve both."""
    if not config.speedrun:
        return build_adamw(list(model.parameters()), config)
    block_matrices = [p for block in model.blocks for p in block.parameters() if p.dim() == 2]
    embedding = model.embedding.weight
    muon_ids = {id(parameter) for parameter in block_matrices}
    others = [p for p in model.parameters() if id(p) not in muon_ids and p is not embedding]
    muon = BatchedMuon(
        block_matrices,
        lr=config.lr,
        weight_decay=config.weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = build_adamw(others, config)
    # embedding: lr x10 as in modded-nanogpt, weight_decay /10 so lr * weight_decay is unchanged
    adamw.add_param_group(
        {
            "params": [embedding],
            "lr": config.lr * EMBEDDING_LR_SCALE,
            "weight_decay": config.weight_decay / EMBEDDING_LR_SCALE,
        }
    )
    return MuonWithAdamW([muon, adamw])


class LanguageModel(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = GPT(config)
        if config.compile:
            # compile the loss, not the model, so the fp32 cast and cross-entropy fuse into the
            # graph; GPT.generate still calls the uncompiled forward, and state-dict keys stay clean
            self.per_sequence_loss = torch.compile(self.per_sequence_loss)

    def per_sequence_loss(
        self, tokens: Int[torch.Tensor, "batch seq"], targets: Int[torch.Tensor, "batch seq"]
    ) -> Float[torch.Tensor, " batch"]:
        logits = self.model(tokens)
        losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
        return losses.view(targets.shape).mean(dim=1)

    def on_train_batch_start(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        if self.config.speedrun:
            momentum = schedule.muon_momentum_at(self.config, self.global_step)
            for group in self.trainer.optimizers[0].param_groups:
                if "ns_coefficients" in group:  # Muon's groups; AdamW's have betas instead
                    group["momentum"] = momentum

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        return self.per_sequence_loss(batch["tokens"], batch["targets"]).mean()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> dict:
        return {
            "per_sequence_loss": self.per_sequence_loss(batch["tokens"], batch["targets"]),
            "source_id": batch["source_id"],
        }

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.config)
        lr_scale = LambdaLR(
            optimizer, lambda step: schedule.lr_at(self.config, step) / self.config.lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scale, "interval": "step", "name": "lr"},
        }
