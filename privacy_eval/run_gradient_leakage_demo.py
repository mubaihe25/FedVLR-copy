"""Run a lightweight gradient leakage demo summary.

This runner is intentionally synthetic and bounded. It produces a
``gradient_leakage_summary.json``-compatible payload using tiny tensor inputs.
It is not a full FedVLR image reconstruction pipeline, DLG implementation, or
InvertingGrad integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from privacy_eval.gradient_leakage_probe import GradientLeakageProbe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny synthetic gradient leakage demo probe."
    )
    parser.add_argument(
        "--demo-kind",
        choices=["tensor", "image"],
        default="tensor",
        help="Synthetic demo input shape.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=8,
        help="Height for the tiny synthetic tensor/image demo.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=8,
        help="Width for the tiny synthetic tensor/image demo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic synthetic noise.",
    )
    parser.add_argument(
        "--sensitive-modality",
        choices=["image", "multimodal", "unknown"],
        default="image",
        help="Modality label included in the summary.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for writing gradient_leakage_summary.json.",
    )
    return parser


def build_synthetic_gradient(kind: str, height: int, width: int, seed: int) -> Dict[str, Any]:
    height = max(1, min(int(height), 32))
    width = max(1, min(int(width), 32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if kind == "image":
        base = torch.linspace(0.0, 1.0, steps=height * width, dtype=torch.float32)
        gradient = base.reshape(1, height, width)
        gradient = gradient + torch.normal(
            mean=0.0,
            std=0.01,
            size=gradient.shape,
            generator=generator,
        )
        return {"synthetic_image_encoder.grad": gradient}
    gradient = torch.normal(
        mean=0.0,
        std=0.2,
        size=(1, height, width),
        generator=generator,
    )
    return {"synthetic_tensor.grad": gradient}


def run_demo(
    demo_kind: str = "tensor",
    height: int = 8,
    width: int = 8,
    seed: int = 42,
    sensitive_modality: str = "image",
) -> Dict[str, Any]:
    gradient_source = build_synthetic_gradient(demo_kind, height, width, seed)
    probe = GradientLeakageProbe(
        config={
            "sensitive_modality": sensitive_modality,
            "norm_reference": 10.0,
            "energy_reference": 100.0,
        }
    )
    summary = probe.evaluate_gradients(gradient_source)
    summary["status"] = "available"
    summary["demo_only"] = True
    summary["not_from_real_training"] = True
    summary["input_source"] = "synthetic_{}_gradient".format(demo_kind)
    summary["note"] = (
        "demo probe only; not full FedVLR image reconstruction, DLG, or InvertingGrad"
    )
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    summary = run_demo(
        demo_kind=args.demo_kind,
        height=args.height,
        width=args.width,
        seed=args.seed,
        sensitive_modality=args.sensitive_modality,
    )
    if args.output_json:
        write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
