import lightning as L
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch.optim.lr_scheduler import LambdaLR

from . import schedule
from .config import Config
from .model import GPT


def build_adamw(parameters: list[torch.nn.Parameter], config: Config) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only."""
    decayed = [parameter for parameter in parameters if parameter.dim() >= 2]
    undecayed = [parameter for parameter in parameters if parameter.dim() < 2]
    groups = [
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.95), eps=1e-8)


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


def build_optimizer(model: GPT, config: Config) -> torch.optim.Optimizer:
    """AdamW; with config.speedrun, Muon on the matrices inside the blocks and AdamW on the rest
    (embedding, head, norms). Muon's "match_rms_adamw" scales its update to AdamW's typical RMS,
    so one config.lr and one schedule serve both."""
    if not config.speedrun:
        return build_adamw(list(model.parameters()), config)
    block_matrices = [p for block in model.blocks for p in block.parameters() if p.dim() == 2]
    muon_ids = {id(parameter) for parameter in block_matrices}
    others = [p for p in model.parameters() if id(p) not in muon_ids]
    muon = torch.optim.Muon(
        block_matrices,
        lr=config.lr,
        weight_decay=config.weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    return MuonWithAdamW([muon, build_adamw(others, config)])


class LanguageModel(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = GPT(config)
        if config.compile:
            self.model.compile()  # in place, so state-dict keys stay clean

    def per_sequence_loss(
        self, tokens: Int[torch.Tensor, "batch seq"], targets: Int[torch.Tensor, "batch seq"]
    ) -> Float[torch.Tensor, " batch"]:
        logits = self.model(tokens)
        losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
        return losses.view(targets.shape).mean(dim=1)

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
