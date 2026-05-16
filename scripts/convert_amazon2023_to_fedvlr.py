"""Convert a local Amazon Reviews 2023 category shard into FedVLR format.

The converter is intentionally offline-only: it reads local review/metadata
JSONL files, stores image URLs as metadata, and creates URL-hash placeholder
image features. It does not download Amazon data or images.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


USER_FIELD = "userID"
ITEM_FIELD = "itemID"
SPLIT_FIELD = "split_label"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert local Amazon Reviews 2023 reviews/meta files into a FedVLR PoC dataset."
    )
    parser.add_argument("--reviews-file", required=True, help="All_Beauty reviews jsonl/json/jsonl.gz.")
    parser.add_argument("--meta-file", required=True, help="All_Beauty metadata jsonl/json/jsonl.gz.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--max-interactions", type=int, default=50000)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=3)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--image-dim", type=int, default=128)
    parser.add_argument("--max-similarity-items", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8-sig", errors="ignore")


def iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from JSONL, JSON array, or JSON object with a data list."""
    with open_text(path) as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            payload = json.load(handle)
            for record in payload:
                if isinstance(record, dict):
                    yield record
            return
        if first == "{":
            text = handle.read().strip()
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records = payload.get("data") or payload.get("records") or payload.get("reviews")
                    if isinstance(records, list):
                        for record in records:
                            if isinstance(record, dict):
                                yield record
                    else:
                        yield payload
                    return
            except json.JSONDecodeError:
                handle.seek(0)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def first_present(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def item_raw_id(record: Dict[str, Any]) -> Optional[str]:
    value = first_present(record, ("parent_asin", "asin", "item_id", "itemID"))
    return None if value in (None, "") else str(value)


def flatten_strings(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        output: List[str] = []
        for item in value.values():
            output.extend(flatten_strings(item))
        return output
    if isinstance(value, (list, tuple, set)):
        output = []
        for item in value:
            output.extend(flatten_strings(item))
        return output
    return [str(value)]


def normalize_categories(value: Any) -> List[str]:
    seen = set()
    categories: List[str] = []
    for text in flatten_strings(value):
        text = str(text).strip()
        if text and text not in seen:
            seen.add(text)
            categories.append(text)
    return categories


def extract_image_url(images: Any) -> Optional[str]:
    if images in (None, ""):
        return None
    if isinstance(images, str):
        return images or None
    if isinstance(images, dict):
        for key in ("hi_res", "large", "large_image_url", "medium", "medium_image_url", "thumb", "small_image_url", "url"):
            value = images.get(key)
            if isinstance(value, list):
                for item in value:
                    if item:
                        return str(item)
            elif value:
                return str(value)
        for value in images.values():
            found = extract_image_url(value)
            if found:
                return found
    if isinstance(images, list):
        for item in images:
            found = extract_image_url(item)
            if found:
                return found
    return None


def read_metadata(path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    for record in iter_json_records(path):
        raw_item_id = item_raw_id(record)
        if raw_item_id is None:
            continue
        categories = normalize_categories(
            first_present(record, ("categories", "category", "cat", "breadcrumb"))
        )
        main_category = first_present(record, ("main_category", "main_cat", "category"))
        if isinstance(main_category, list):
            main_category = main_category[0] if main_category else None
        description_parts = flatten_strings(first_present(record, ("description", "details", "about_product")))
        feature_parts = flatten_strings(first_present(record, ("features", "feature", "bullet_point")))
        metadata[raw_item_id] = {
            "raw_item_id": raw_item_id,
            "title": first_present(record, ("title", "name")) or "",
            "description": " ".join(str(part).strip() for part in description_parts if str(part).strip()),
            "features": [str(part).strip() for part in feature_parts if str(part).strip()],
            "categories": categories,
            "main_category": str(main_category).strip() if main_category not in (None, "") else "",
            "image_url": extract_image_url(first_present(record, ("images", "image", "image_url", "img_url"))),
            "brand_or_store": first_present(record, ("store", "brand", "manufacturer")) or "",
        }
    if not metadata:
        warnings.append("metadata file contained no records with parent_asin/asin")
    return metadata, warnings


def read_reviews(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    skipped = 0
    usable = 0
    latest_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for order, record in enumerate(iter_json_records(path)):
        raw_user_id = first_present(record, ("user_id", "userID", "reviewerID", "profile_id"))
        raw_item_id = item_raw_id(record)
        rating = first_present(record, ("rating", "overall", "score"))
        timestamp = first_present(record, ("timestamp", "unixReviewTime", "time"))
        if raw_user_id in (None, "") or raw_item_id in (None, "") or rating in (None, ""):
            skipped += 1
            continue
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            skipped += 1
            continue
        try:
            timestamp_value = int(float(timestamp)) if timestamp not in (None, "") else order
        except (TypeError, ValueError):
            timestamp_value = order
        row = {
            "raw_user_id": str(raw_user_id),
            "raw_item_id": str(raw_item_id),
            "rating": rating_value,
            "timestamp": timestamp_value,
            "_order": order,
        }
        usable += 1
        key = (row["raw_user_id"], row["raw_item_id"])
        old = latest_by_pair.get(key)
        if old is None or (row["timestamp"], row["_order"]) >= (old["timestamp"], old["_order"]):
            latest_by_pair[key] = row
    rows = list(latest_by_pair.values())
    duplicate_count = max(0, usable - len(rows))
    if skipped:
        warnings.append("skipped {} review rows with missing user/item/rating".format(skipped))
    if duplicate_count:
        warnings.append("deduplicated {} repeated user-item reviews by latest timestamp".format(duplicate_count))
    if not rows:
        warnings.append("reviews file contained no usable rows")
    return rows, warnings


def enforce_min_counts(
    rows: Sequence[Dict[str, Any]],
    min_user_interactions: int,
    min_item_interactions: int,
    max_rounds: int = 20,
) -> List[Dict[str, Any]]:
    filtered = list(rows)
    min_user_interactions = max(3, int(min_user_interactions))
    min_item_interactions = max(1, int(min_item_interactions))
    for _ in range(max_rounds):
        user_counts = Counter(row["raw_user_id"] for row in filtered)
        item_counts = Counter(row["raw_item_id"] for row in filtered)
        next_rows = [
            row
            for row in filtered
            if user_counts[row["raw_user_id"]] >= min_user_interactions
            and item_counts[row["raw_item_id"]] >= min_item_interactions
        ]
        if len(next_rows) == len(filtered):
            return next_rows
        filtered = next_rows
    return filtered


def crop_rows(
    rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    max_users: int,
    max_items: int,
    max_interactions: int,
    min_user_interactions: int,
    min_item_interactions: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    filtered = enforce_min_counts(rows, min_user_interactions, min_item_interactions)
    if len(filtered) < len(rows):
        warnings.append("filtered {} rows by min user/item interactions".format(len(rows) - len(filtered)))

    if max_items and len({row["raw_item_id"] for row in filtered}) > max_items:
        item_counts = Counter(row["raw_item_id"] for row in filtered)
        selected_items = {
            item_id
            for item_id, _ in sorted(
                item_counts.items(),
                key=lambda item: (
                    0 if item[0] in metadata else 1,
                    -item[1],
                    item[0],
                ),
            )[:max_items]
        }
        filtered = [row for row in filtered if row["raw_item_id"] in selected_items]
        filtered = enforce_min_counts(filtered, min_user_interactions, min_item_interactions)
        warnings.append("cropped to at most {} items, preferring metadata-rich active items".format(max_items))

    if max_users and len({row["raw_user_id"] for row in filtered}) > max_users:
        user_counts = Counter(row["raw_user_id"] for row in filtered)
        selected_users = {
            user_id
            for user_id, _ in sorted(user_counts.items(), key=lambda item: (-item[1], item[0]))[:max_users]
        }
        filtered = [row for row in filtered if row["raw_user_id"] in selected_users]
        filtered = enforce_min_counts(filtered, min_user_interactions, min_item_interactions)
        warnings.append("cropped to at most {} active users".format(max_users))

    if max_interactions and len(filtered) > max_interactions:
        user_counts = Counter(row["raw_user_id"] for row in filtered)
        selected_users = set()
        running = 0
        for user_id, count in sorted(user_counts.items(), key=lambda item: (-item[1], item[0])):
            if selected_users and running + count > max_interactions:
                continue
            selected_users.add(user_id)
            running += count
            if running >= max_interactions:
                break
        filtered = [row for row in filtered if row["raw_user_id"] in selected_users]
        filtered = enforce_min_counts(filtered, min_user_interactions, min_item_interactions)
        warnings.append("cropped to at most {} interactions via active-user subset".format(max_interactions))

    final_users = {row["raw_user_id"] for row in filtered}
    final_items = {row["raw_item_id"] for row in filtered}
    missing_metadata = sum(1 for item_id in final_items if item_id not in metadata)
    if missing_metadata:
        warnings.append("{} retained items have no metadata record".format(missing_metadata))
    if any(Counter(row["raw_user_id"] for row in filtered)[user_id] < 3 for user_id in final_users):
        warnings.append("some retained users have fewer than 3 interactions; split may be incomplete")
    return filtered, warnings


def build_mappings(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    user_counts = Counter(row["raw_user_id"] for row in rows)
    item_counts = Counter(row["raw_item_id"] for row in rows)
    user_mapping = {
        raw_id: index
        for index, raw_id in enumerate(
            [raw_id for raw_id, _ in sorted(user_counts.items(), key=lambda item: (-item[1], item[0]))]
        )
    }
    item_mapping = {
        raw_id: index
        for index, raw_id in enumerate(
            [raw_id for raw_id, _ in sorted(item_counts.items(), key=lambda item: (-item[1], item[0]))]
        )
    }
    return user_mapping, item_mapping


def split_rows(
    rows: Sequence[Dict[str, Any]],
    user_mapping: Dict[str, int],
    item_mapping: Dict[str, int],
) -> List[Dict[str, Any]]:
    per_user: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        per_user.setdefault(row["raw_user_id"], []).append(row)
    output: List[Dict[str, Any]] = []
    for raw_user_id, user_rows in per_user.items():
        ordered = sorted(user_rows, key=lambda row: (row["timestamp"], row["_order"], row["raw_item_id"]))
        for index, row in enumerate(ordered):
            if index == len(ordered) - 1:
                split_label = 2
            elif index == len(ordered) - 2:
                split_label = 1
            else:
                split_label = 0
            output.append(
                {
                    USER_FIELD: user_mapping[row["raw_user_id"]],
                    ITEM_FIELD: item_mapping[row["raw_item_id"]],
                    "rating": float(row["rating"]),
                    "timestamp": int(row["timestamp"]),
                    SPLIT_FIELD: split_label,
                }
            )
    return sorted(output, key=lambda row: (row[USER_FIELD], row["timestamp"], row[ITEM_FIELD]))


def tokens_from_text(text: str) -> Iterator[str]:
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token:
            yield token


def stable_hash(text: str, salt: str = "") -> int:
    digest = hashlib.blake2b((salt + text).encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    non_zero = norms[:, 0] > 0
    matrix[non_zero] = matrix[non_zero] / norms[non_zero]
    return matrix


def hash_text_feature(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens_from_text(text):
        value = stable_hash(token, "text")
        index = value % dim
        sign = 1.0 if ((value >> 8) & 1) else -1.0
        vector[index] += sign
    return vector


def hash_string_feature(text: Optional[str], dim: int, salt: str) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    if not text:
        return vector
    for chunk_start in range(0, len(text), 16):
        chunk = text[chunk_start : chunk_start + 16]
        value = stable_hash(chunk, salt)
        index = value % dim
        sign = 1.0 if ((value >> 8) & 1) else -1.0
        vector[index] += sign
    return vector


def metadata_text(meta: Dict[str, Any]) -> str:
    parts = [
        meta.get("title") or "",
        meta.get("description") or "",
        " ".join(flatten_strings(meta.get("features"))),
        " ".join(flatten_strings(meta.get("categories"))),
        meta.get("main_category") or "",
        meta.get("brand_or_store") or "",
    ]
    return " ".join(str(part) for part in parts if part)


def build_metadata_items(
    item_mapping: Dict[str, int],
    metadata: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for raw_item_id, item_id in sorted(item_mapping.items(), key=lambda item: item[1]):
        meta = metadata.get(raw_item_id, {})
        categories = list(meta.get("categories") or [])
        main_category = meta.get("main_category") or ""
        target_group = main_category or (categories[0] if categories else "")
        items.append(
            {
                "itemID": int(item_id),
                "raw_item_id": raw_item_id,
                "title": meta.get("title") or "",
                "description": meta.get("description") or "",
                "features": list(meta.get("features") or []),
                "categories": categories,
                "main_category": main_category,
                "image_url": meta.get("image_url"),
                "brand_or_store": meta.get("brand_or_store") or "",
                "target_group": target_group,
            }
        )
    return items


def build_features(
    items: Sequence[Dict[str, Any]],
    text_dim: int,
    image_dim: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    text_features = np.zeros((len(items), text_dim), dtype=np.float32)
    image_features = np.zeros((len(items), image_dim), dtype=np.float32)
    image_url_count = 0
    for item in items:
        item_id = int(item["itemID"])
        text_features[item_id] = hash_text_feature(metadata_text(item), text_dim)
        if item.get("image_url"):
            image_url_count += 1
            image_features[item_id] = hash_string_feature(str(item["image_url"]), image_dim, "image_url")
    return l2_normalize(text_features), l2_normalize(image_features), {"items_with_image_url": image_url_count}


def build_similarity(
    text_features: np.ndarray,
    max_similarity_items: int,
    warnings: List[str],
) -> np.ndarray:
    item_count = int(text_features.shape[0])
    limit = min(item_count, max(1, int(max_similarity_items)))
    if limit < item_count:
        warnings.append(
            "similarity.npy contains only the first {} items because item count {} exceeds max_similarity_items".format(
                limit,
                item_count,
            )
        )
    subset = text_features[:limit].astype(np.float32, copy=False)
    return np.matmul(subset, subset.T).astype(np.float32)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def dataset_profile(
    output_rows: Sequence[Dict[str, Any]],
    users: int,
    items: int,
    source_reviews: Path,
    source_meta: Path,
    feature_stats: Dict[str, int],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    split_counts = Counter(int(row[SPLIT_FIELD]) for row in output_rows)
    interactions = len(output_rows)
    sparsity = None
    if users and items:
        sparsity = float(1.0 - interactions / float(users * items))
    return {
        "dataset_name": "AMAZON_BEAUTY_POC",
        "source": {
            "name": "Amazon Reviews 2023 All_Beauty local files",
            "reviews_file": str(source_reviews),
            "meta_file": str(source_meta),
        },
        "users": int(users),
        "items": int(items),
        "interactions": int(interactions),
        "train_interactions": int(split_counts.get(0, 0)),
        "valid_interactions": int(split_counts.get(1, 0)),
        "test_interactions": int(split_counts.get(2, 0)),
        "sparsity": sparsity,
        "modalities": ["interaction", "text", "image_url_placeholder", "category_metadata"],
        "text_feature_method": "deterministic hashing bag-of-words over title, description, features, categories, main_category, and brand/store",
        "image_feature_method": "url-hash placeholder; not visual embeddings and no image files were downloaded",
        "similarity_method": "text_features cosine similarity",
        "items_with_image_url": int(feature_stats.get("items_with_image_url", 0)),
        "warnings": list(warnings),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def convert_amazon2023_to_fedvlr(args: argparse.Namespace) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError(
            "numpy is required for feature generation; install project requirements or run inside .venv"
        )
    reviews_path = Path(args.reviews_file)
    meta_path = Path(args.meta_file)
    output_dir = Path(args.output_dir)
    if args.text_dim <= 0 or args.image_dim <= 0:
        raise ValueError("text_dim and image_dim must be positive")

    random.seed(args.seed)
    warnings: List[str] = []
    metadata, metadata_warnings = read_metadata(meta_path)
    review_rows, review_warnings = read_reviews(reviews_path)
    warnings.extend(metadata_warnings)
    warnings.extend(review_warnings)

    cropped_rows, crop_warnings = crop_rows(
        review_rows,
        metadata,
        max_users=args.max_users,
        max_items=args.max_items,
        max_interactions=args.max_interactions,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    warnings.extend(crop_warnings)
    if not cropped_rows:
        raise ValueError("No interactions remain after filtering; lower min-counts or max limits")

    user_mapping, item_mapping = build_mappings(cropped_rows)
    inter_rows = split_rows(cropped_rows, user_mapping, item_mapping)
    metadata_items = build_metadata_items(item_mapping, metadata)
    text_features, image_features, feature_stats = build_features(
        metadata_items,
        text_dim=args.text_dim,
        image_dim=args.image_dim,
    )
    similarity = build_similarity(text_features, args.max_similarity_items, warnings)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "inter.csv",
        inter_rows,
        [USER_FIELD, ITEM_FIELD, "rating", "timestamp", SPLIT_FIELD],
    )
    write_csv(
        output_dir / "u_id_mapping.csv",
        [
            {"userID": user_id, "raw_user_id": raw_user_id}
            for raw_user_id, user_id in sorted(user_mapping.items(), key=lambda item: item[1])
        ],
        ["userID", "raw_user_id"],
    )
    write_csv(
        output_dir / "i_id_mapping.csv",
        [
            {"itemID": item_id, "raw_item_id": raw_item_id}
            for raw_item_id, item_id in sorted(item_mapping.items(), key=lambda item: item[1])
        ],
        ["itemID", "raw_item_id"],
    )
    write_json(
        output_dir / "item_metadata.json",
        {
            "metadata_type": "item_metadata",
            "dataset": "AMAZON_BEAUTY_POC",
            "items": metadata_items,
            "warnings": [
                warning for warning in warnings if "metadata" in warning.lower()
            ],
            "note": "image_url stores the remote URL only; images were not downloaded",
        },
    )
    np.save(output_dir / "text_features.npy", text_features)
    np.save(output_dir / "image_features.npy", image_features)
    np.save(output_dir / "similarity.npy", similarity)
    profile = dataset_profile(
        inter_rows,
        users=len(user_mapping),
        items=len(item_mapping),
        source_reviews=reviews_path,
        source_meta=meta_path,
        feature_stats=feature_stats,
        warnings=warnings
        + ["image_features.npy are URL-hash placeholders, not visual embeddings"],
    )
    write_json(output_dir / "dataset_profile.json", profile)
    return profile


def main() -> int:
    args = build_parser().parse_args()
    profile = convert_amazon2023_to_fedvlr(args)
    print(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
