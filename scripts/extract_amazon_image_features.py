from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = Path("datasets/AMAZON_BEAUTY_POC/item_image_manifest.json")
DEFAULT_OUTPUT = Path("datasets/AMAZON_BEAUTY_POC/image_features_extracted.npy")
DEFAULT_MAPPING = Path("datasets/AMAZON_BEAUTY_POC/image_features_extracted_items.json")
DEFAULT_SUMMARY = Path("datasets/AMAZON_BEAUTY_POC/image_feature_extraction_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract optional Amazon Beauty image features from cached images."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-npy", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", default="resnet18", choices=["resnet18"])
    parser.add_argument(
        "--weights",
        default="none",
        choices=["none", "default"],
        help="Use none by default to avoid implicit dependency/model downloads.",
    )
    return parser


def repo_relative(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def unavailable_summary(
    output_json: Path,
    reason: str,
    manifest: Path,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    summary = {
        "summary_type": "image_feature_extraction_summary",
        "dataset": "AMAZON_BEAUTY_POC",
        "feature_extraction_available": False,
        "torchvision_available": False,
        "model": None,
        "feature_file": None,
        "item_mapping_file": None,
        "item_count": 0,
        "failed_count": 0,
        "source_manifest": repo_relative(manifest),
        "warnings": warnings or [reason],
        "note": (
            "No feature file was generated. image_features.npy remains a URL-hash "
            "placeholder and was not modified."
        ),
    }
    write_json(output_json, summary)
    return summary


def import_runtime() -> Tuple[Any, Any, Any, Any, Optional[str]]:
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from torchvision import models, transforms  # type: ignore

        return np, torch, Image, (models, transforms), None
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, str(exc)


def image_entries(manifest: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    entries = manifest.get("items", []) if isinstance(manifest, dict) else []
    result: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        local_path = entry.get("local_image_path")
        if not local_path:
            continue
        if str(entry.get("status", "")) not in {"downloaded", "exists"}:
            continue
        result.append(entry)
        if len(result) >= limit:
            break
    return result


def build_model(torch: Any, models: Any, weights_mode: str) -> Tuple[Any, bool, List[str]]:
    warnings: List[str] = []
    weights = None
    pretrained = False
    if weights_mode == "default":
        try:
            weights = models.ResNet18_Weights.DEFAULT
            pretrained = True
        except Exception as exc:  # noqa: BLE001
            warnings.append("ResNet18 default weights unavailable: {}".format(exc))
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, pretrained, warnings


def run_extraction(
    manifest_path: Path,
    output_npy: Path,
    output_mapping: Path,
    output_json: Path,
    limit: int,
    model_name: str,
    weights_mode: str,
) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        return unavailable_summary(output_json, "image manifest not found or invalid", manifest_path)
    np, torch, Image, runtime, import_error = import_runtime()
    if import_error:
        return unavailable_summary(
            output_json,
            "torch/torchvision/PIL runtime unavailable: {}".format(import_error),
            manifest_path,
        )
    models, transforms = runtime
    model, pretrained, warnings = build_model(torch, models, weights_mode)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    features: List[Any] = []
    mapping: List[Dict[str, Any]] = []
    failed = 0
    for entry in image_entries(manifest, limit):
        path = ROOT / str(entry.get("local_image_path"))
        try:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                vector = model(tensor).squeeze(0).detach().cpu().numpy()
            features.append(vector)
            mapping.append(
                {
                    "itemID": str(entry.get("itemID")),
                    "raw_item_id": entry.get("raw_item_id"),
                    "local_image_path": entry.get("local_image_path"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            warnings.append("item {} extraction failed: {}".format(entry.get("itemID"), exc))
    if not features:
        return unavailable_summary(
            output_json,
            "no cached images could be processed",
            manifest_path,
            warnings=warnings or ["no cached images could be processed"],
        )
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_mapping.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, np.stack(features))
    output_mapping.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "summary_type": "image_feature_extraction_summary",
        "dataset": "AMAZON_BEAUTY_POC",
        "feature_extraction_available": True,
        "torchvision_available": True,
        "model": model_name,
        "weights": weights_mode,
        "pretrained_weights": pretrained,
        "feature_file": repo_relative(output_npy),
        "item_mapping_file": repo_relative(output_mapping),
        "item_count": len(features),
        "failed_count": failed,
        "source_manifest": repo_relative(manifest_path),
        "warnings": warnings,
        "note": (
            "Optional cached-image feature sidecar. It does not replace "
            "datasets/AMAZON_BEAUTY_POC/image_features.npy; if weights=none, "
            "features are visual tensor smoke features rather than semantic embeddings."
        ),
    }
    write_json(output_json, summary)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    summary = run_extraction(
        manifest_path=Path(args.manifest),
        output_npy=Path(args.output_npy),
        output_mapping=Path(args.output_mapping),
        output_json=Path(args.output_json),
        limit=args.limit,
        model_name=args.model,
        weights_mode=args.weights,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
