"""Attack extension skeletons for the FedVLR project."""

from attacks.base_attack import BaseAttack
from attacks.client_preference_leakage_probe import ClientPreferenceLeakageProbe
from attacks.client_update_scale_attack import ClientUpdateScaleAttack
from attacks.model_replacement_attack import ModelReplacementAttack
from attacks.sign_flip_attack import SignFlipAttack
from attacks.noop_attack import NoOpAttack
from attacks.registry import get_attack, list_attacks, register_attack

__all__ = [
    "BaseAttack",
    "ClientPreferenceLeakageProbe",
    "ClientUpdateScaleAttack",
    "ModelReplacementAttack",
    "SignFlipAttack",
    "NoOpAttack",
    "register_attack",
    "get_attack",
    "list_attacks",
]
