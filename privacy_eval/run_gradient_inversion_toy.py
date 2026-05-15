"""Run a tiny DLG-style gradient inversion toy demo.

This runner uses only a synthetic tensor and a very small neural network. It
matches dummy gradients to target gradients for a bounded number of steps. It is
not a FedVLR image reconstruction pipeline and does not copy external DLG or
InvertingGradients code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn


class TinyInversionModel(nn.Module):
    """A bounded tiny model for gradient matching smoke tests."""

    def __init__(self, input_shape: Sequence[int], num_classes: int = 3) -> None:
        super().__init__()
        input_dim = int(torch.tensor(list(input_shape)).prod().item())
        hidden_dim = min(32, max(4, input_dim // 8))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def psnr_like_from_mse(mse: float) -> float:
    return float(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))


def risk_level_from_mse(mse: float) -> str:
    if mse <= 0.02:
        return "high"
    if mse <= 0.10:
        return "medium"
    return "low"


def tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach().float().cpu()
    return {
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "numel": int(detached.numel()),
    }


def run_gradient_inversion_toy(
    channels: int = 1,
    height: int = 16,
    width: int = 16,
    steps: int = 50,
    seed: int = 42,
    lr: float = 0.15,
    output_dir: Optional[Path] = None,
    save_tensors: bool = False,
) -> Dict[str, Any]:
    channels = max(1, min(int(channels), 3))
    height = max(4, min(int(height), 32))
    width = max(4, min(int(width), 32))
    steps = max(1, min(int(steps), 200))
    torch.manual_seed(int(seed))

    input_shape: Tuple[int, int, int] = (channels, height, width)
    model = TinyInversionModel(input_shape)
    model.eval()
    label = torch.tensor([1], dtype=torch.long)
    original = torch.rand((1, *input_shape), dtype=torch.float32)

    target_logits = model(original)
    target_loss = nn.functional.cross_entropy(target_logits, label)
    target_grads = torch.autograd.grad(target_loss, tuple(model.parameters()))
    target_grads = [gradient.detach() for gradient in target_grads]

    dummy = torch.rand_like(original, requires_grad=True)
    optimizer = torch.optim.Adam([dummy], lr=float(lr))
    final_matching_loss = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        dummy_logits = model(dummy)
        dummy_loss = nn.functional.cross_entropy(dummy_logits, label)
        dummy_grads = torch.autograd.grad(
            dummy_loss,
            tuple(model.parameters()),
            create_graph=True,
        )
        matching_loss = sum(
            torch.mean((dummy_grad - target_grad) ** 2)
            for dummy_grad, target_grad in zip(dummy_grads, target_grads)
        )
        matching_loss.backward()
        optimizer.step()
        with torch.no_grad():
            dummy.clamp_(0.0, 1.0)
        final_matching_loss = float(matching_loss.detach().item())

    reconstruction_mse = float(torch.mean((dummy.detach() - original) ** 2).item())
    psnr_like = psnr_like_from_mse(reconstruction_mse)
    summary = {
        "probe_type": "gradient_inversion_toy",
        "method_family": "DLG-style toy gradient matching",
        "input_shape": list(input_shape),
        "steps": steps,
        "final_gradient_matching_loss": final_matching_loss,
        "reconstruction_mse": reconstruction_mse,
        "psnr_like": psnr_like,
        "risk_level": risk_level_from_mse(reconstruction_mse),
        "demo_only": True,
        "not_from_real_training": True,
        "not_full_fedvlr_reconstruction": True,
        "no_external_code_copied": True,
        "original_tensor_stats": tensor_stats(original),
        "reconstructed_tensor_stats": tensor_stats(dummy),
        "note": (
            "toy gradient inversion demo for leakage-risk education; not full "
            "FedVLR original image recovery or InvertingGrad integration"
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "gradient_inversion_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
        if save_tensors:
            original_path = output_dir / "original_tensor.pt"
            reconstructed_path = output_dir / "reconstructed_tensor.pt"
            torch.save(original.detach().cpu(), original_path)
            torch.save(dummy.detach().cpu(), reconstructed_path)
            summary["original_tensor_path"] = str(original_path)
            summary["reconstructed_tensor_path"] = str(reconstructed_path)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny gradient inversion toy demo.")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-tensors", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_gradient_inversion_toy(
        channels=args.channels,
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        lr=args.lr,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        save_tensors=bool(args.save_tensors),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
