from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_METADATA = Path("datasets/AMAZON_BEAUTY_POC/item_metadata.json")
DEFAULT_INTER = Path("datasets/AMAZON_BEAUTY_POC/inter.csv")
DEFAULT_TARGETS = Path("outputs/security_sidecars/AMAZON_BEAUTY_POC/target_items.json")
DEFAULT_OUTPUT_DIR = Path("datasets/AMAZON_BEAUTY_POC/item_images")
DEFAULT_MANIFEST = Path("datasets/AMAZON_BEAUTY_POC/item_image_manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache Amazon Beauty PoC item images without changing model feature files."
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--inter-file", default=str(DEFAULT_INTER))
    parser.add_argument("--target-items-json", default=str(DEFAULT_TARGETS))
    parser.add_argument("--recommendation-comparison", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Optional[Path]) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def metadata_items(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    items = payload.get("items") if isinstance(payload, dict) else payload
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("itemID", item.get("item_id", item.get("id")))
            if item_id is not None:
                result[str(item_id)] = dict(item)
    elif isinstance(items, dict):
        for key, value in items.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("itemID", key)
                result[str(key)] = item
    return result


def normalize_item_id(value: Any) -> Optional[str]:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def collect_ids(obj: Any, keys: Sequence[str]) -> Set[str]:
    ids: Set[str] = set()
    normalized_keys = {key.lower() for key in keys}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in normalized_keys:
                values = value if isinstance(value, list) else [value]
                for item in values:
                    item_id = normalize_item_id(item)
                    if item_id is not None:
                        ids.add(item_id)
            ids.update(collect_ids(value, keys))
    elif isinstance(obj, list):
        for value in obj:
            ids.update(collect_ids(value, keys))
            if not isinstance(value, (dict, list)):
                item_id = normalize_item_id(value)
                if item_id is not None:
                    ids.add(item_id)
    return ids


def read_target_ids(path: Path) -> Set[str]:
    payload = read_json(path)
    if payload is None:
        return set()
    if isinstance(payload, dict) and "target_items" in payload:
        return collect_ids(payload["target_items"], ["item_id", "itemID", "id"])
    return collect_ids(payload, ["item_id", "itemID", "id"])


def read_recommendation_ids(paths: Sequence[str]) -> Set[str]:
    ids: Set[str] = set()
    keys = [
        "item_id",
        "itemID",
        "recommended_item_id",
        "baseline_item_id",
        "attack_item_id",
        "defense_item_id",
    ]
    for raw_path in paths:
        payload = read_json(Path(raw_path))
        if payload is not None:
            ids.update(collect_ids(payload, keys))
    return ids


def detect_delimiter(path: Path) -> str:
    try:
        first_line = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[0]
    except Exception:
        return ","
    return "\t" if "\t" in first_line and "," not in first_line else ","


def interaction_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=detect_delimiter(path))
            for row in reader:
                item_id = normalize_item_id(
                    row.get("itemID") or row.get("item_id") or row.get("item")
                )
                if item_id is not None:
                    counts[item_id] += 1
    except Exception:
        return Counter()
    return counts


def priority_order(
    metadata: Dict[str, Dict[str, Any]],
    target_ids: Set[str],
    recommendation_ids: Set[str],
    counts: Counter[str],
) -> List[str]:
    def sort_key(item_id: str) -> Tuple[int, int, int, str]:
        return (
            0 if item_id in target_ids else 1,
            0 if item_id in recommendation_ids else 1,
            -counts.get(item_id, 0),
            item_id,
        )

    return sorted(metadata.keys(), key=sort_key)


def image_extension(url: str, content_type: Optional[str]) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return guessed
    return ".jpg"


def download_image(url: str, output_stem: Path, timeout: float) -> Tuple[str, Optional[Path], Optional[str]]:
    request = Request(
        url,
        headers={
            "User-Agent": "FedVLR-image-cache/1.0 (+local research smoke)",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - public metadata URLs.
            content_type = response.headers.get("Content-Type")
            data = response.read()
    except URLError as exc:
        return "failed", None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return "failed", None, str(exc)
    if not data:
        return "failed", None, "empty response"
    output_path = output_stem.with_suffix(image_extension(url, content_type))
    try:
        output_path.write_bytes(data)
    except Exception as exc:  # noqa: BLE001
        return "failed", None, str(exc)
    return "downloaded", output_path, None


def cache_images(
    metadata_path: Path,
    inter_file: Path,
    target_items_json: Path,
    recommendation_comparisons: Sequence[str],
    output_dir: Path,
    manifest_path: Path,
    limit: int,
    timeout: float,
    sleep_seconds: float,
) -> Dict[str, Any]:
    metadata = metadata_items(metadata_path)
    target_ids = read_target_ids(target_items_json)
    recommendation_ids = read_recommendation_ids(recommendation_comparisons)
    counts = interaction_counts(inter_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    selected_item_ids = priority_order(metadata, target_ids, recommendation_ids, counts)[: max(0, limit)]
    for item_id in selected_item_ids:
        item = metadata.get(item_id, {})
        image_url = item.get("image_url") or item.get("image") or item.get("img_url")
        raw_item_id = item.get("raw_item_id") or item.get("parent_asin") or item.get("asin")
        entry: Dict[str, Any] = {
            "itemID": item_id,
            "raw_item_id": raw_item_id,
            "image_url": image_url,
            "local_image_path": None,
            "status": "skipped_no_url",
            "warning": None,
            "priority": {
                "target_item": item_id in target_ids,
                "recommendation_item": item_id in recommendation_ids,
                "interaction_count": counts.get(item_id, 0),
            },
        }
        if not image_url:
            entry["warning"] = "missing image_url"
            entries.append(entry)
            continue
        existing = sorted(output_dir.glob("{}.*".format(item_id)))
        if existing:
            entry.update(
                {
                    "local_image_path": repo_relative(existing[0]),
                    "status": "exists",
                    "warning": None,
                }
            )
            entries.append(entry)
            continue
        status, output_path, warning = download_image(str(image_url), output_dir / str(item_id), timeout)
        entry["status"] = status
        entry["warning"] = warning
        if output_path is not None:
            entry["local_image_path"] = repo_relative(output_path)
        entries.append(entry)
        if warning:
            warnings.append("item {}: {}".format(item_id, warning))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    counts_by_status = Counter(entry["status"] for entry in entries)
    manifest = {
        "summary_type": "amazon_item_image_cache",
        "dataset": "AMAZON_BEAUTY_POC",
        "metadata_file": repo_relative(metadata_path),
        "inter_file": repo_relative(inter_file),
        "target_items_json": repo_relative(target_items_json),
        "recommendation_comparison_files": [
            repo_relative(Path(path)) for path in recommendation_comparisons
        ],
        "output_dir": repo_relative(output_dir),
        "limit": limit,
        "selected_item_count": len(entries),
        "downloaded_count": counts_by_status.get("downloaded", 0),
        "existing_count": counts_by_status.get("exists", 0),
        "failed_count": counts_by_status.get("failed", 0),
        "skipped_count": counts_by_status.get("skipped_no_url", 0),
        "items": entries,
        "warnings": warnings,
        "note": (
            "Local image cache only. This script does not modify image_features.npy "
            "and cached images must not be described as FedVLR visual embeddings."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = build_parser().parse_args()
    manifest = cache_images(
        metadata_path=Path(args.metadata),
        inter_file=Path(args.inter_file),
        target_items_json=Path(args.target_items_json),
        recommendation_comparisons=args.recommendation_comparison,
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
        limit=args.limit,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
