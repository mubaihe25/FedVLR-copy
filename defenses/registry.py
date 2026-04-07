"""Minimal registry for defense skeletons."""

from __future__ import annotations

from typing import Dict, List, Type

from defenses.base_defense import BaseDefense

_DEFENSE_REGISTRY: Dict[str, Type[BaseDefense]] = {}


def register_defense(name: str, defense_cls: Type[BaseDefense]) -> None:
    """Register a defense class by name."""
    _DEFENSE_REGISTRY[name] = defense_cls


def get_defense(name: str) -> Type[BaseDefense]:
    """Return a registered defense class."""
    if name not in _DEFENSE_REGISTRY:
        raise KeyError("Unknown defense: {}".format(name))
    return _DEFENSE_REGISTRY[name]


def list_defenses() -> List[str]:
    """List all registered defense names."""
    return sorted(_DEFENSE_REGISTRY.keys())
