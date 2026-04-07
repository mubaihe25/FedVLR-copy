"""No-op defense placeholder."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

from defenses.base_defense import BaseDefense
from defenses.registry import register_defense


class NoOpDefense(BaseDefense):
    """Defense placeholder that does not modify any training state."""

    def __init__(self, name: str = "noop", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, config=config)

    def before_round(
        self, round_state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        round_state.setdefault("defense_placeholders", []).append(self.name)
        return round_state


register_defense("noop", NoOpDefense)
register_defense("noopdefense", NoOpDefense)
