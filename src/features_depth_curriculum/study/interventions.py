"""B: interventions, changes to the model or to the prompts (metrics.md, Study matrix names)."""

from typing import Literal

from .interface import Context, Intervention, Subject, register


@register
class NoIntervention(Intervention):
    """Leave the model as it is: the reference for every other intervention."""

    name: Literal["none"] = "none"
    kind = "none"

    def apply(self, subject: Subject, context: Context) -> Subject:
        return subject
