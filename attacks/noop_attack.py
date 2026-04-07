"""No-op attack placeholder."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

from attacks.base_attack import BaseAttack
from attacks.registry import register_attack


class NoOpAttack(BaseAttack):
    """Attack placeholder that does not modify any training state."""

    def __init__(self, name: str = "noop", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, config=config)

    def before_round(
        self, round_state: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        round_state.setdefault("attack_placeholders", []).append(self.name)
        return round_state


register_attack("noop", NoOpAttack)
register_attack("noopattack", NoOpAttack)
