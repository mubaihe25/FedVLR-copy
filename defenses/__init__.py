"""Defense extension skeletons for the FedVLR project."""

from defenses.base_defense import BaseDefense
from defenses.noop_defense import NoOpDefense
from defenses.registry import get_defense, list_defenses, register_defense

__all__ = [
    "BaseDefense",
    "NoOpDefense",
    "register_defense",
    "get_defense",
    "list_defenses",
]
