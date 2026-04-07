"""Attack extension skeletons for the FedVLR project."""

from attacks.base_attack import BaseAttack
from attacks.noop_attack import NoOpAttack
from attacks.registry import get_attack, list_attacks, register_attack

__all__ = [
    "BaseAttack",
    "NoOpAttack",
    "register_attack",
    "get_attack",
    "list_attacks",
]
