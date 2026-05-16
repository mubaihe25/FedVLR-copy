"""Attack extension skeletons for the FedVLR project."""

from attacks.base_attack import BaseAttack
from attacks.target_interaction_injection import TargetInteractionInjectionPlanner
from attacks.noop_attack import NoOpAttack
from attacks.registry import get_attack, list_attacks, register_attack

try:
    from attacks.client_preference_leakage_probe import ClientPreferenceLeakageProbe
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    ClientPreferenceLeakageProbe = None  # type: ignore[assignment]

try:
    from attacks.client_update_scale_attack import ClientUpdateScaleAttack
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    ClientUpdateScaleAttack = None  # type: ignore[assignment]

try:
    from attacks.model_replacement_attack import ModelReplacementAttack
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    ModelReplacementAttack = None  # type: ignore[assignment]

try:
    from attacks.poisoning_attack import PoisoningAttack
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    PoisoningAttack = None  # type: ignore[assignment]

try:
    from attacks.sign_flip_attack import SignFlipAttack
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    SignFlipAttack = None  # type: ignore[assignment]

try:
    from attacks.targeted_poisoning_attack import TargetedPoisoningAttack
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    TargetedPoisoningAttack = None  # type: ignore[assignment]

__all__ = [
    "BaseAttack",
    "ClientPreferenceLeakageProbe",
    "ClientUpdateScaleAttack",
    "ModelReplacementAttack",
    "PoisoningAttack",
    "SignFlipAttack",
    "TargetInteractionInjectionPlanner",
    "TargetedPoisoningAttack",
    "NoOpAttack",
    "register_attack",
    "get_attack",
    "list_attacks",
]
