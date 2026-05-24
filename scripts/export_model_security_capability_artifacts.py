from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    ROOT
    / "outputs"
    / "model_security_capability_matrix"
    / "model_security_capability_matrix.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "showcase_artifacts" / "model_security_capability_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model security capability matrix artifacts for showcase/API readers."
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix)
    output_dir = Path(args.output_dir)
    matrix = load_json(matrix_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []

    matrix_artifact = output_dir / "model_security_capability_matrix.json"
    write_json(matrix_artifact, matrix)
    files.append(matrix_artifact)

    supported = {
        "summary_type": "model_security_supported_demos",
        "generated_at": utc_now(),
        "source_matrix": rel(matrix_path),
        "supported_demos": matrix.get("supported_demos", []),
    }
    supported_path = output_dir / "supported_demos.json"
    write_json(supported_path, supported)
    files.append(supported_path)

    unsupported = {
        "summary_type": "model_security_unsupported_reasons",
        "generated_at": utc_now(),
        "source_matrix": rel(matrix_path),
        "unsupported_reasons": matrix.get("unsupported_reasons", []),
    }
    unsupported_path = output_dir / "unsupported_reasons.json"
    write_json(unsupported_path, unsupported)
    files.append(unsupported_path)

    labels = {
        "summary_type": "model_security_frontend_labels",
        "generated_at": utc_now(),
        "source_matrix": rel(matrix_path),
        "recommended_frontend_labels": matrix.get("recommended_frontend_labels", {}),
        "status_counts": matrix.get("status_counts", {}),
        "warnings": matrix.get("warnings", []),
    }
    labels_path = output_dir / "recommended_frontend_labels.json"
    write_json(labels_path, labels)
    files.append(labels_path)

    manifest = {
        "summary_type": "model_security_capability_showcase_manifest",
        "generated_at": utc_now(),
        "source_matrix": rel(matrix_path),
        "artifact_files": [rel(path) for path in files],
        "file_count": len(files),
        "warnings": matrix.get("warnings", []),
    }
    manifest_path = output_dir / "showcase_manifest.json"
    write_json(manifest_path, manifest)
    files.append(manifest_path)

    print(
        json.dumps(
            {
                "output_dir": rel(output_dir),
                "file_count": len(files),
                "artifact_files": [rel(path) for path in files],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
