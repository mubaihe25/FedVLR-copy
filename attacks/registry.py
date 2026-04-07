"""Minimal registry for attack skeletons."""

from __future__ import annotations

from typing import Dict, List, Type

from attacks.base_attack import BaseAttack

_ATTACK_REGISTRY: Dict[str, Type[BaseAttack]] = {}


def register_attack(name: str, attack_cls: Type[BaseAttack]) -> None:
    """Register an attack class by name."""
    _ATTACK_REGISTRY[name] = attack_cls


def get_attack(name: str) -> Type[BaseAttack]:
    """Return a registered attack class."""
    if name not in _ATTACK_REGISTRY:
        raise KeyError("Unknown attack: {}".format(name))
    return _ATTACK_REGISTRY[name]


def list_attacks() -> List[str]:
    """List all registered attack names."""
    return sorted(_ATTACK_REGISTRY.keys())
