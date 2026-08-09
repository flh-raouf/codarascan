#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a strict declared-dependency manifest from a wheel or sdist."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def metadata(artifact: Path) -> bytes:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            name = next(item for item in archive.namelist() if item.endswith(".dist-info/METADATA"))
            return archive.read(name)
    with tarfile.open(artifact, "r:gz") as archive:
        info = next(item for item in archive.getmembers() if item.name.endswith("/PKG-INFO"))
        extracted = archive.extractfile(info)
        if extracted is None:
            raise RuntimeError(f"unable to read metadata from {artifact}")
        return extracted.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    raw = metadata(arguments.artifact)
    parsed = BytesParser().parsebytes(raw)
    manifest = {
        "schema_version": 1,
        "artifact": arguments.artifact.name,
        "artifact_sha256": hashlib.sha256(arguments.artifact.read_bytes()).hexdigest(),
        "distribution": parsed["Name"],
        "version": parsed["Version"],
        "license_expression": parsed["License-Expression"],
        "requires_python": parsed["Requires-Python"],
        "declared_dependencies": sorted(parsed.get_all("Requires-Dist", [])),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"dependency manifest written: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
