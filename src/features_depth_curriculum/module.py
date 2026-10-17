import lightning as L
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch.optim.lr_scheduler import LambdaLR

from . import schedule
from .config import Config
from .model import GPT


def build_optimizer(model: GPT, config: Config) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only."""
    decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    undecayed = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    groups = [
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.95), eps=1e-8)


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
