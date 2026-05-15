"""Check whether Opacus is available for future DP-SGD work.

This script intentionally does not install Opacus and does not attach
PrivacyEngine to FedVLR. It reports environment feasibility and the expected
training-loop requirements for a future formal DP-SGD branch.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _module_version(module_name: str) -> tuple[bool, Optional[str], Optional[str]]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - feasibility should report, not fail.
        return False, None, str(exc)
    return True, str(getattr(module, "__version__", "unknown")), None


def run_check() -> Dict[str, Any]:
    torch_available, torch_version, torch_error = _module_version("torch")
    opacus_available, opacus_version, opacus_error = _module_version("opacus")
    summary = {
        "check_type": "opacus_feasibility",
        "torch_available": torch_available,
        "torch_version": torch_version,
        "torch_error": torch_error,
        "opacus_available": opacus_available,
        "opacus_version": opacus_version,
        "opacus_error": opacus_error,
        "requires_training_loop_refactor": True,
        "requires_per_sample_gradient": True,
        "requires_optimizer_or_privacy_engine_integration": True,
        "recommended_status": "future_work",
        "formal_dp_currently_enabled": False,
        "note": (
            "Opacus feasibility only; FedVLR dp_noise remains central DP-style "
            "update noise without a formal privacy accountant"
        ),
    }
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Opacus availability without modifying training.")
    parser.add_argument("--output-json", help="Optional opacus_feasibility_summary.json path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_check()
    if args.output_json:
        write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
