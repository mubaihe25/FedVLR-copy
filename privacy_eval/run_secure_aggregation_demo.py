"""Executable secure aggregation mask-cancellation demo.

This is a synthetic demonstration only. It is not a production cryptographic
secure aggregation protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a synthetic secure aggregation demo.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--client-count", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dropout-client-ids", nargs="*", default=[])
    return parser


def vector_norm(values: Sequence[float]) -> float:
    return float(math.sqrt(sum(float(value) * float(value) for value in values)))


def add(left: Sequence[float], right: Sequence[float]) -> List[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def sub(left: Sequence[float], right: Sequence[float]) -> List[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(int(args.seed))
    client_count = max(2, int(args.client_count))
    dimension = max(1, int(args.dimension))
    updates = {
        str(client_id): [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        for client_id in range(client_count)
    }
    dropout = {str(item) for item in args.dropout_client_ids}
    active_clients = [client_id for client_id in updates if client_id not in dropout]
    pairwise_masks: Dict[str, List[float]] = {}
    masked_updates = {client_id: list(update) for client_id, update in updates.items()}
    mask_count = 0
    for left_index, left_id in enumerate(updates):
        for right_id in list(updates)[left_index + 1 :]:
            mask = [rng.uniform(-0.5, 0.5) for _ in range(dimension)]
            pairwise_masks["{}::{}".format(left_id, right_id)] = mask
            mask_count += 1
            masked_updates[left_id] = add(masked_updates[left_id], mask)
            masked_updates[right_id] = sub(masked_updates[right_id], mask)

    aggregate_plain = [0.0 for _ in range(dimension)]
    aggregate_masked = [0.0 for _ in range(dimension)]
    for client_id in active_clients:
        aggregate_plain = add(aggregate_plain, updates[client_id])
        aggregate_masked = add(aggregate_masked, masked_updates[client_id])
    residual = sub(aggregate_masked, aggregate_plain)
    dropout_simulation = {
        "enabled": bool(dropout),
        "dropout_client_ids": sorted(dropout),
        "active_client_count": len(active_clients),
        "warning": (
            "Pairwise masks only cancel for all clients; dropout requires a recovery protocol."
            if dropout
            else None
        ),
    }
    return {
        "summary_type": "secure_aggregation_demo_summary",
        "defense_type": "secure_aggregation_demo",
        "client_count": client_count,
        "active_client_count": len(active_clients),
        "dimension": dimension,
        "pairwise_mask_count": mask_count,
        "aggregate_residual_norm": vector_norm(residual),
        "masked_individual_updates_visible": False,
        "dropout_simulation": dropout_simulation,
        "demo_only": True,
        "not_production_cryptographic_protocol": True,
        "warnings": [dropout_simulation["warning"]] if dropout_simulation["warning"] else [],
        "note": "Synthetic pairwise mask cancellation demo; not PySyft and not production secure aggregation.",
    }


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
