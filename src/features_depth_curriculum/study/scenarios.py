"""C: scenarios, how the facts got into the model (design.md, Scenarios)."""

from typing import Literal

from ..model import GPT
from ..pilot.run import load_model
from .interface import Context, Scenario, register


@register
class Checkpoint(Scenario):
    """A model that is already trained: `baseline`, `seed-twin` and every injection schedule are
    trained runs, so each one is this scenario with its own experiment config."""

    name: Literal["checkpoint"] = "checkpoint"

    def build(self, context: Context) -> GPT:
        return load_model(context.config, self.checkpoint, context.device, context.smoke)
