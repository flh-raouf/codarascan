#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete or contaminated CodaraScan wheel/sdist artifacts."""

from __future__ import annotations

import argparse
import re
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from glob import glob
from pathlib import Path, PurePosixPath

FORBIDDEN = re.compile(
    r"(^|/)(archive|benchmarks/datasets|docs/barcode-report|reports|tests|third_party|"
    r"__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.venv|dist|build|"
    r"\.env(?:\..*)?|credentials?|secrets?)(/|$)",
    re.IGNORECASE,
)
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".DS_Store", ".pem", ".key")
SENSITIVE_CONTENT = re.compile(
    rb"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|private evaluation document"
)
LOCAL_PATH = re.compile(rb"/Users/[^/\x00]+/|/home/[^/\x00]+/|[A-Za-z]:\\Users\\")
MAX_MEMBER_BYTES = 10 * 1024 * 1024


def project_version() -> str:
    """Return the release version declared by the source checkout."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def expand_artifacts(patterns: list[Path]) -> list[Path]:
    """Expand shell patterns even when the invoking shell leaves them literal."""
    artifacts: list[Path] = []
    for pattern in patterns:
        value = str(pattern)
        if any(character in value for character in "*?["):
            matches = sorted(Path(match) for match in glob(value))
            if not matches:
                raise SystemExit(f"artifact pattern matched no files: {pattern}")
            artifacts.extend(matches)
        else:
            artifacts.append(pattern)
    return artifacts


def names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path, "r:gz") as archive:
        return archive.getnames()


def member_payloads(path: Path) -> list[tuple[str, bytes]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [
                (info.filename, archive.read(info))
                for info in archive.infolist()
                if not info.is_dir() and info.file_size <= MAX_MEMBER_BYTES
            ]
    with tarfile.open(path, "r:gz") as archive:
        output = []
        for info in archive.getmembers():
            if not info.isfile() or info.size > MAX_MEMBER_BYTES:
                continue
            extracted = archive.extractfile(info)
            if extracted is not None:
                output.append((info.name, extracted.read()))
        return output


def member_sizes(path: Path) -> list[tuple[str, int]]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [
                (info.filename, info.file_size) for info in archive.infolist() if not info.is_dir()
            ]
    with tarfile.open(path, "r:gz") as archive:
        return [(info.name, info.size) for info in archive.getmembers() if info.isfile()]


def artifact_metadata(path: Path) -> bytes:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            name = next(
                member for member in archive.namelist() if member.endswith(".dist-info/METADATA")
            )
            return archive.read(name)
    with tarfile.open(path, "r:gz") as archive:
        info = next(member for member in archive.getmembers() if member.name.endswith("/PKG-INFO"))
        extracted = archive.extractfile(info)
        if extracted is None:
            raise RuntimeError(f"unable to read package metadata from {path}")
        return extracted.read()


def inspect(path: Path) -> None:
    members = names(path)
    lowered = [member.lower() for member in members]
    required_fragments = [
        "license",
        "notice",
        "third_party_notices.md",
        "codarascan/schemas/result.schema.json",
        "codarascan/schemas/stream.schema.json",
        "codarascan/schemas/worker.schema.json",
        "codarascan/schemas/result.decoded.example.json",
        "codarascan/schemas/result.localized.example.json",
        "codarascan/schemas/stream.example.json",
        "codarascan/py.typed",
        "sttg_native.cpp",
    ]
    if path.suffix == ".whl":
        required_fragments.append("entry_points.txt")
    else:
        required_fragments.extend(("/pyproject.toml", "/setup.py"))
    missing = [
        fragment
        for fragment in required_fragments
        if not any(fragment in member for member in lowered)
    ]
    forbidden = [
        member
        for member in members
        if FORBIDDEN.search(member) or member.endswith(FORBIDDEN_SUFFIXES)
    ]
    unsafe_paths = [
        member
        for member in members
        if member.startswith("/") or ".." in PurePosixPath(member).parts
    ]
    oversized = [member for member, size in member_sizes(path) if size > MAX_MEMBER_BYTES]
    sensitive = []
    local_paths = []
    missing_spdx = []
    for member, payload in member_payloads(path):
        if SENSITIVE_CONTENT.search(payload):
            sensitive.append(member)
        if LOCAL_PATH.search(payload):
            local_paths.append(member)
        if (
            member.endswith((".py", ".cpp"))
            and b"SPDX-License-Identifier: Apache-2.0" not in payload[:1024]
        ):
            missing_spdx.append(member)
    if path.suffix == ".whl" and not any(
        "codarascan/_sttg_native" in member and member.endswith((".so", ".pyd", ".dylib"))
        for member in members
    ):
        missing.append("compiled codarascan._sttg_native")
    parsed = BytesParser().parsebytes(artifact_metadata(path))
    metadata_errors = []
    expected_metadata = {
        "Name": "codarascan",
        "Version": project_version(),
        "License-Expression": "Apache-2.0",
        "Requires-Python": "<3.15,>=3.11",
    }
    for key, expected in expected_metadata.items():
        if parsed[key] != expected:
            metadata_errors.append(f"{key}={parsed[key]!r}, expected {expected!r}")
    if (
        missing
        or forbidden
        or unsafe_paths
        or oversized
        or sensitive
        or local_paths
        or missing_spdx
        or metadata_errors
    ):
        raise SystemExit(
            f"artifact inspection failed for {path}: missing={missing}, "
            f"forbidden={forbidden[:20]}, sensitive={sensitive[:20]}, "
            f"unsafe_paths={unsafe_paths[:20]}, oversized={oversized[:20]}, "
            f"local_paths={local_paths[:20]}, missing_spdx={missing_spdx[:20]}, "
            f"metadata={metadata_errors}"
        )
    print(f"artifact inspection passed: {path} ({len(members)} members)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    arguments = parser.parse_args()
    for artifact in expand_artifacts(arguments.artifacts):
        inspect(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
