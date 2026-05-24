"""Run a tiny Opacus DP-SGD demo when Opacus is installed.

This is deliberately not wired into the FedVLR training loop. It demonstrates
the formal-DP tooling path and reports availability when the dependency is
missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an optional Opacus toy DP demo.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    return parser


def unavailable_summary(reason: str, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "summary_type": "opacus_toy_summary",
        "metric_type": "opacus_toy_demo",
        "opacus_available": False,
        "demo_only": True,
        "fedvlr_training_loop_integrated": False,
        "epsilon": None,
        "delta": args.delta,
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "sample_count": max(8, int(args.sample_count)),
        "seed": int(args.seed),
        "warnings": [reason],
        "note": "Opacus toy demo is a formal-DP route demonstration, not full FedVLR DP.",
    }


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import torch
        from opacus import PrivacyEngine
    except Exception as exc:  # noqa: BLE001
        return unavailable_summary("opacus or torch unavailable: {}".format(exc), args)

    torch.manual_seed(int(args.seed))
    sample_count = max(8, int(args.sample_count))
    x = torch.randn(sample_count, 4)
    y = (torch.sum(x, dim=1) > 0).long()
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    privacy_engine = PrivacyEngine()
    try:
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        losses = []
        for _ in range(max(1, int(args.epochs))):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
        epsilon = float(privacy_engine.get_epsilon(delta=args.delta))
        return {
            "summary_type": "opacus_toy_summary",
            "metric_type": "opacus_toy_demo",
            "opacus_available": True,
            "demo_only": True,
            "fedvlr_training_loop_integrated": False,
            "epsilon": epsilon,
            "delta": args.delta,
            "noise_multiplier": args.noise_multiplier,
            "max_grad_norm": args.max_grad_norm,
            "sample_count": sample_count,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "final_loss": losses[-1] if losses else None,
            "warnings": [],
            "note": "Formal DP accountant from Opacus toy model; not full FedVLR DP.",
        }
    except Exception as exc:  # noqa: BLE001
        return unavailable_summary("opacus toy run failed: {}".format(exc), args)


def main() -> int:
    args = build_parser().parse_args()
    payload = run_demo(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
