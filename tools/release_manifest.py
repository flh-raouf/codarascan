#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic SHA-256 evidence for a complete release artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-artifacts", type=int)
    arguments = parser.parse_args()

    artifacts = sorted(arguments.artifacts, key=lambda path: path.name)
    if len({path.name for path in artifacts}) != len(artifacts):
        raise SystemExit("release artifact filenames must be unique")
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise SystemExit(f"release artifacts do not exist: {missing}")
    if arguments.expected_artifacts is not None and len(artifacts) != arguments.expected_artifacts:
        raise SystemExit(
            f"expected {arguments.expected_artifacts} release artifacts, received {len(artifacts)}"
        )

    manifest = {
        "schema_version": 1,
        "distribution": "codarascan",
        "version": "0.1.0",
        "artifact_count": len(artifacts),
        "artifacts": [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifacts
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"release manifest written: {arguments.output} ({len(artifacts)} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
