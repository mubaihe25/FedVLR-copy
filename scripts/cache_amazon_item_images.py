from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_METADATA = Path("datasets/AMAZON_BEAUTY_POC/item_metadata.json")
DEFAULT_INTER = Path("datasets/AMAZON_BEAUTY_POC/inter.csv")
DEFAULT_TARGETS = Path("outputs/security_sidecars/AMAZON_BEAUTY_POC/target_items.json")
DEFAULT_V25_RECOMMENDATIONS = Path(
    "outputs/showcase_artifacts/amazon_beauty_poc_v25_backend_smoke/recommendation_comparison.json"
)
DEFAULT_SHOWCASE_ROOT = Path("outputs/showcase_artifacts")
DEFAULT_OUTPUT_DIR = Path("datasets/AMAZON_BEAUTY_POC/item_images")
DEFAULT_THUMB_DIR = DEFAULT_OUTPUT_DIR / "thumbs"
DEFAULT_MANIFEST = Path("datasets/AMAZON_BEAUTY_POC/item_image_manifest.json")

RECOMMENDATION_KEYS = {
    "item_id",
    "itemid",
    "itemID".lower(),
    "recommended_item_id",
    "baseline_item_id",
    "attack_item_id",
    "defense_item_id",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache bounded Amazon Beauty PoC display images and thumbnails."
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--inter-file", default=str(DEFAULT_INTER))
    parser.add_argument("--target-items-json", default=str(DEFAULT_TARGETS))
    parser.add_argument("--v25-recommendation-comparison", default=str(DEFAULT_V25_RECOMMENDATIONS))
    parser.add_argument("--showcase-root", default=str(DEFAULT_SHOWCASE_ROOT))
    parser.add_argument("--recommendation-comparison", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--thumb-dir", default=str(DEFAULT_THUMB_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--thumb-size", type=int, default=224)
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


def normalize_item_id(value: Any) -> Optional[str]:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def first_text(item: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def category_text(item: Dict[str, Any]) -> Optional[str]:
    direct = first_text(item, ["category", "item_category", "main_category"])
    if direct:
        return direct
    categories = item.get("categories")
    if isinstance(categories, list):
        flattened = []
        for value in categories:
            if isinstance(value, str) and value.strip():
                flattened.append(value.strip())
            elif isinstance(value, list):
                flattened.extend(str(part).strip() for part in value if str(part).strip())
        if flattened:
            return " / ".join(flattened[:3])
    return None


def metadata_items(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    items = payload.get("items") if isinstance(payload, dict) else payload
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = normalize_item_id(item.get("itemID", item.get("item_id", item.get("id"))))
            if item_id is not None:
                result[item_id] = dict(item)
    elif isinstance(items, dict):
        for key, value in items.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("itemID", key)
                result[str(key)] = item
    return result


def collect_ids(obj: Any, keys: Sequence[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    normalized_keys = {key.lower() for key in keys}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in normalized_keys:
                values = value if isinstance(value, list) else [value]
                for item in values:
                    item_id = normalize_item_id(item)
                    if item_id is not None:
                        counts[item_id] += 1
            counts.update(collect_ids(value, keys))
    elif isinstance(obj, list):
        for value in obj:
            counts.update(collect_ids(value, keys))
    return counts


def read_target_ids(path: Path) -> set[str]:
    payload = read_json(path)
    if payload is None:
        return set()
    if isinstance(payload, list):
        return {item_id for item_id in (normalize_item_id(item) for item in payload) if item_id is not None}
    if isinstance(payload, dict) and "target_items" in payload:
        target_items = payload["target_items"]
        if isinstance(target_items, list) and all(not isinstance(item, (dict, list)) for item in target_items):
            return {item_id for item_id in (normalize_item_id(item) for item in target_items) if item_id is not None}
        return set(collect_ids(payload["target_items"], ["item_id", "itemID", "id"]).keys())
    return set(collect_ids(payload, ["item_id", "itemID", "id"]).keys())


def recommendation_files(showcase_root: Path, explicit_paths: Sequence[str], v25_path: Path) -> List[Path]:
    files: List[Path] = []
    if v25_path.exists():
        files.append(v25_path)
    for raw_path in explicit_paths:
        path = Path(raw_path)
        if path.exists():
            files.append(path)
    if showcase_root.exists():
        files.extend(sorted(showcase_root.glob("*/recommendation_comparison.json")))
    seen: set[Path] = set()
    unique_files: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)
    return unique_files


def read_recommendation_counts(paths: Sequence[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        payload = read_json(path)
        if payload is not None:
            counts.update(collect_ids(payload, list(RECOMMENDATION_KEYS)))
    return counts


def read_v25_recommendation_ids(path: Path) -> set[str]:
    payload = read_json(path)
    if payload is None:
        return set()
    return set(collect_ids(payload, list(RECOMMENDATION_KEYS)).keys())


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
    target_ids: set[str],
    v25_recommendation_ids: set[str],
    recommendation_counts: Counter[str],
    counts: Counter[str],
) -> List[str]:
    def sort_key(item_id: str) -> Tuple[int, int, int, int, str]:
        return (
            0 if item_id in target_ids else 1,
            0 if item_id in v25_recommendation_ids else 1,
            -recommendation_counts.get(item_id, 0),
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


def existing_image(output_dir: Path, item_id: str) -> Optional[Path]:
    candidates = [
        path
        for path in sorted(output_dir.glob(f"{item_id}.*"))
        if path.is_file() and path.parent.name != "thumbs"
    ]
    return candidates[0] if candidates else None


def create_thumbnail(source_path: Optional[Path], thumb_dir: Path, item_id: str, size: int) -> Tuple[Optional[Path], Optional[str]]:
    if source_path is None or not source_path.exists():
        return None, "source image missing"
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None, "Pillow unavailable"
    try:
        thumb_dir.mkdir(parents=True, exist_ok=True)
        output_path = thumb_dir / f"{item_id}.jpg"
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            image.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), (18, 24, 38))
            offset = ((size - image.width) // 2, (size - image.height) // 2)
            canvas.paste(image, offset)
            canvas.save(output_path, format="JPEG", quality=82, optimize=True)
        return output_path, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def iter_selected_items(values: Iterable[str], limit: int) -> List[str]:
    result: List[str] = []
    for item_id in values:
        if item_id not in result:
            result.append(item_id)
        if len(result) >= max(0, limit):
            break
    return result


def cache_images(
    metadata_path: Path,
    inter_file: Path,
    target_items_json: Path,
    v25_recommendation_comparison: Path,
    showcase_root: Path,
    recommendation_comparisons: Sequence[str],
    output_dir: Path,
    thumb_dir: Path,
    manifest_path: Path,
    limit: int,
    timeout: float,
    sleep_seconds: float,
    thumb_size: int,
) -> Dict[str, Any]:
    metadata = metadata_items(metadata_path)
    target_ids = read_target_ids(target_items_json)
    rec_files = recommendation_files(showcase_root, recommendation_comparisons, v25_recommendation_comparison)
    v25_recommendation_ids = read_v25_recommendation_ids(v25_recommendation_comparison)
    recommendation_counts = read_recommendation_counts(rec_files)
    counts = interaction_counts(inter_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    selected_item_ids = iter_selected_items(
        priority_order(metadata, target_ids, v25_recommendation_ids, recommendation_counts, counts),
        limit,
    )

    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    thumbnail_available = True

    for item_id in selected_item_ids:
        item = metadata.get(item_id, {})
        image_url = first_text(item, ["image_url", "image", "img_url"])
        raw_item_id = first_text(item, ["raw_item_id", "parent_asin", "asin", "item_asin"])
        local_path: Optional[Path] = existing_image(output_dir, item_id)
        status = "exists" if local_path else "skipped_no_url"
        warning: Optional[str] = None
        thumbnail_path: Optional[Path] = None
        thumbnail_warning: Optional[str] = None

        if not image_url:
            warning = "missing image_url"
        elif local_path is None:
            status, local_path, warning = download_image(str(image_url), output_dir / str(item_id), timeout)
        if local_path is not None:
            thumbnail_path, thumbnail_warning = create_thumbnail(local_path, thumb_dir, item_id, thumb_size)
            if thumbnail_warning == "Pillow unavailable":
                thumbnail_available = False
            if thumbnail_warning and not warning:
                warning = thumbnail_warning

        entry: Dict[str, Any] = {
            "item_id": item_id,
            "itemID": item_id,
            "raw_item_id": raw_item_id,
            "title": first_text(item, ["title", "item_title", "name"]),
            "category": category_text(item),
            "image_url": image_url,
            "local_image_path": repo_relative(local_path) if local_path else None,
            "thumbnail_path": repo_relative(thumbnail_path) if thumbnail_path else None,
            "status": status,
            "warning": warning,
            "priority": {
                "target_item": item_id in target_ids,
                "v25_recommendation_item": item_id in v25_recommendation_ids,
                "showcase_recommendation_count": recommendation_counts.get(item_id, 0),
                "interaction_count": counts.get(item_id, 0),
            },
        }
        entries.append(entry)
        if warning:
            warnings.append(f"item {item_id}: {warning}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    counts_by_status = Counter(entry["status"] for entry in entries)
    thumbnail_count = sum(1 for entry in entries if entry.get("thumbnail_path"))
    manifest = {
        "summary_type": "amazon_item_image_cache",
        "dataset": "AMAZON_BEAUTY_POC",
        "metadata_file": repo_relative(metadata_path),
        "inter_file": repo_relative(inter_file),
        "target_items_json": repo_relative(target_items_json),
        "v25_recommendation_comparison": repo_relative(v25_recommendation_comparison),
        "recommendation_comparison_files": [repo_relative(path) for path in rec_files],
        "output_dir": repo_relative(output_dir),
        "thumbnail_dir": repo_relative(thumb_dir),
        "limit": limit,
        "selected_item_count": len(entries),
        "downloaded_count": counts_by_status.get("downloaded", 0),
        "existing_count": counts_by_status.get("exists", 0),
        "failed_count": counts_by_status.get("failed", 0),
        "skipped_count": counts_by_status.get("skipped_no_url", 0),
        "thumbnail_count": thumbnail_count,
        "thumbnail_available": thumbnail_available,
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
        v25_recommendation_comparison=Path(args.v25_recommendation_comparison),
        showcase_root=Path(args.showcase_root),
        recommendation_comparisons=args.recommendation_comparison,
        output_dir=Path(args.output_dir),
        thumb_dir=Path(args.thumb_dir),
        manifest_path=Path(args.manifest),
        limit=args.limit,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        thumb_size=args.thumb_size,
    )
    summary = {
        "selected_item_count": manifest["selected_item_count"],
        "downloaded_count": manifest["downloaded_count"],
        "existing_count": manifest["existing_count"],
        "failed_count": manifest["failed_count"],
        "skipped_count": manifest["skipped_count"],
        "thumbnail_count": manifest["thumbnail_count"],
        "thumbnail_available": manifest["thumbnail_available"],
        "manifest": repo_relative(Path(args.manifest)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
