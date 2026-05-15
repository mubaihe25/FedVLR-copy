"""Build security sidecar files for FedVLR experiments.

Generated files are auxiliary artifacts for probes/exporters. They do not alter
training outputs and do not fabricate semantic item metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from privacy_eval.generate_membership_labels import (
    detect_delimiter,
    generate_membership_labels,
    read_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build FedVLR security sidecars for membership, target items, and item metadata stubs."
    )
    parser.add_argument("--dataset", default="KU")
    parser.add_argument("--data-root", help="Dataset root. Defaults to ./datasets.")
    parser.add_argument("--output-dir", help="Defaults to outputs/security_sidecars/<dataset>.")
    parser.add_argument("--recommend-topk-file")
    parser.add_argument("--max-members", type=int, default=1000)
    parser.add_argument("--max-non-members", type=int, default=1000)
    parser.add_argument("--target-item-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def dataset_dir(dataset: str, data_root: Optional[str]) -> Path:
    root = Path(data_root) if data_root else ROOT / "datasets"
    return root / dataset


def build_target_items(
    dataset: str,
    data_root: Optional[str],
    target_item_count: int,
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    inter_path = dataset_dir(dataset, data_root) / "inter.csv"
    rows = read_rows(inter_path)
    if not rows:
        return {
            "target_items": [],
            "description": "attacker target items",
            "selection_strategy": "not_available",
            "warnings": ["inter.csv could not be parsed"],
        }, ["target_items unavailable because inter.csv could not be parsed"]

    counts: Counter[str] = Counter()
    for row in rows:
        if str(row.get("split_label")) != "0":
            continue
        item_id = row.get("itemID") or row.get("item_id") or row.get("item")
        if item_id not in (None, ""):
            counts[str(item_id)] += 1
    if not counts:
        warnings.append("no train item frequencies found")
    target_items = [
        item_id
        for item_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            : max(0, target_item_count)
        ]
    ]
    return {
        "target_items": target_items,
        "description": "attacker target items selected from high-frequency train items for smoke/demo use",
        "selection_strategy": "high_frequency_train_items",
        "target_item_count": len(target_items),
        "proxy_selection": True,
        "warnings": warnings,
    }, warnings


def build_item_metadata_stub(dataset: str, data_root: Optional[str]) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    mapping_path = dataset_dir(dataset, data_root) / "i_id_mapping.csv"
    rows = read_rows(mapping_path)
    if not rows:
        return {
            "metadata_type": "item_metadata_stub",
            "items": {},
            "warnings": ["i_id_mapping.csv could not be parsed"],
            "note": "stub only; no title/tag semantics are available",
        }, ["item metadata stub unavailable because i_id_mapping.csv could not be parsed"]

    items: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("itemID") or row.get("item_id")
        raw_item_id = row.get("asin") or row.get("raw_item_id")
        if item_id in (None, ""):
            continue
        items[str(item_id)] = {
            "item_id": str(item_id),
            "raw_item_id": None if raw_item_id in (None, "") else str(raw_item_id),
            "title": None,
            "tag": None,
            "group": None,
            "source": "i_id_mapping_stub",
        }
    if not items:
        warnings.append("i_id_mapping.csv did not contain itemID rows")
    return {
        "metadata_type": "item_metadata_stub",
        "items": items,
        "warnings": warnings,
        "note": "stub only; no title/tag semantics are available",
    }, warnings


def build_security_sidecars(
    dataset: str = "KU",
    data_root: Optional[str] = None,
    output_dir: Optional[Path] = None,
    recommend_topk_file: Optional[Path] = None,
    max_members: int = 1000,
    max_non_members: int = 1000,
    target_item_count: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir = output_dir or ROOT / "outputs" / "security_sidecars" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    membership = generate_membership_labels(
        dataset=dataset,
        data_root=data_root,
        recommend_topk_file=recommend_topk_file,
        max_members=max_members,
        max_non_members=max_non_members,
        seed=seed,
    )
    target_items, target_warnings = build_target_items(dataset, data_root, target_item_count)
    item_metadata, metadata_warnings = build_item_metadata_stub(dataset, data_root)

    membership_path = output_dir / "membership_labels.json"
    target_items_path = output_dir / "target_items.json"
    item_metadata_path = output_dir / "item_metadata_stub.json"
    manifest_path = output_dir / "security_sidecar_manifest.json"

    write_json(membership_path, membership)
    write_json(target_items_path, target_items)
    write_json(item_metadata_path, item_metadata)

    manifest = {
        "dataset": dataset,
        "sidecar_type": "security_sidecars",
        "membership_labels": str(membership_path.resolve()),
        "target_items": str(target_items_path.resolve()),
        "item_metadata_stub": str(item_metadata_path.resolve()),
        "sources": {
            "dataset_dir": str(dataset_dir(dataset, data_root).resolve()),
            "recommend_topk_file": str(recommend_topk_file.resolve()) if recommend_topk_file else None,
        },
        "warnings": list(membership.get("warnings", [])) + target_warnings + metadata_warnings,
        "note": "item_metadata_stub has no real title/tag semantics; target_items use a high-frequency proxy unless provided by future experiments",
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.smoke and output_dir is None:
        output_dir = ROOT / "outputs" / "security_sidecars" / "{}_smoke".format(args.dataset)
    manifest = build_security_sidecars(
        dataset=args.dataset,
        data_root=args.data_root,
        output_dir=output_dir,
        recommend_topk_file=Path(args.recommend_topk_file) if args.recommend_topk_file else None,
        max_members=args.max_members,
        max_non_members=args.max_non_members,
        target_item_count=args.target_item_count,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
