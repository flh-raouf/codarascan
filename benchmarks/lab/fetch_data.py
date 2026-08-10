#!/usr/bin/env python3
"""Fetch and verify benchmark inputs without putting large files in Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).with_name("datasets.json")
EXPECTED_SCHEMA = "barcode-benchmark-provenance-v1"
CHUNK_SIZE = 1024 * 1024


class RegistryError(ValueError):
    """Raised when the tracked dataset registry is invalid."""


def _repo_path(value: str) -> Path:
    """Resolve a registry path while keeping it inside the repository."""

    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RegistryError(f"artifact path must be repository-relative: {value!r}")
    root = ROOT.resolve()
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RegistryError(f"artifact path escapes repository: {value!r}") from error
    return resolved


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{field} must be a non-empty string")
    return value


def validate_registry(registry: dict[str, Any]) -> None:
    if registry.get("schema_version") != EXPECTED_SCHEMA:
        raise RegistryError(
            f"unsupported registry schema: {registry.get('schema_version')!r}"
        )
    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise RegistryError("registry must contain a non-empty datasets list")

    dataset_ids: set[str] = set()
    artifact_paths: set[str] = set()
    allowed_formats = {"file", "zip", "tar.gz"}
    allowed_kinds = {"kaggle", "manual", "url"}

    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise RegistryError("each dataset entry must be an object")
        dataset_id = _require_text(dataset.get("id"), "dataset id")
        if dataset_id in dataset_ids:
            raise RegistryError(f"duplicate dataset id: {dataset_id}")
        dataset_ids.add(dataset_id)

        source = dataset.get("source")
        if not isinstance(source, dict):
            raise RegistryError(f"{dataset_id}: source must be an object")
        kind = _require_text(source.get("kind"), f"{dataset_id} source kind")
        if kind not in allowed_kinds:
            raise RegistryError(f"{dataset_id}: unsupported source kind {kind!r}")
        page = _require_text(source.get("page"), f"{dataset_id} source page")
        if not page.startswith("https://"):
            raise RegistryError(f"{dataset_id}: source page must use HTTPS")
        if kind == "kaggle":
            _require_text(source.get("dataset"), f"{dataset_id} Kaggle dataset")
            if not isinstance(source.get("version"), int) or source["version"] <= 0:
                raise RegistryError(f"{dataset_id}: Kaggle version must be positive")

        artifacts = dataset.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RegistryError(f"{dataset_id}: artifacts must be a non-empty list")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise RegistryError(f"{dataset_id}: artifact must be an object")
            path_value = _require_text(artifact.get("path"), f"{dataset_id} artifact path")
            _repo_path(path_value)
            if path_value in artifact_paths:
                raise RegistryError(f"duplicate artifact path: {path_value}")
            artifact_paths.add(path_value)
            digest = _require_text(artifact.get("sha256"), f"{path_value} sha256")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise RegistryError(f"{path_value}: sha256 must be 64 lowercase hex characters")
            artifact_format = artifact.get("format")
            if artifact_format not in allowed_formats:
                raise RegistryError(f"{path_value}: unsupported format {artifact_format!r}")
            if kind == "url":
                url = _require_text(artifact.get("url"), f"{path_value} URL")
                if not url.startswith("https://"):
                    raise RegistryError(f"{path_value}: artifact URL must use HTTPS")
            if kind == "manual" and artifact.get("url"):
                raise RegistryError(f"{path_value}: manual artifacts must not have a URL")


def load_registry() -> dict[str, Any]:
    try:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read {REGISTRY_PATH}: {error}") from error
    if not isinstance(registry, dict):
        raise RegistryError("registry root must be an object")
    validate_registry(registry)
    return registry


def _datasets_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {dataset["id"]: dataset for dataset in registry["datasets"]}


def _dataset(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    dataset = _datasets_by_id(registry).get(dataset_id)
    if dataset is None:
        known = ", ".join(sorted(_datasets_by_id(registry)))
        raise RegistryError(f"unknown dataset {dataset_id!r}; choose one of: {known}")
    return dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(artifact: dict[str, Any]) -> bool:
    path = _repo_path(artifact["path"])
    if not path.is_file():
        print(f"MISSING  {artifact['path']}")
        return False
    actual = sha256(path)
    expected = artifact["sha256"]
    if actual != expected:
        print(f"BAD      {artifact['path']} (expected {expected}, got {actual})")
        return False
    print(f"OK       {artifact['path']}")
    return True


def _download_url(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "barcode-detection-benchmark/1"})
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _download_kaggle(dataset: dict[str, Any], destination: Path) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "the Kaggle CLI is required for this dataset; install it and configure Kaggle credentials"
        )
    source = dataset["source"]
    with tempfile.TemporaryDirectory(prefix="barcode-kaggle-") as temporary:
        command = [
            executable,
            "datasets",
            "download",
            "-d",
            source["dataset"],
            "--dataset-version-number",
            str(source["version"]),
            "-p",
            temporary,
        ]
        print(f"Running Kaggle download for {source['dataset']} version {source['version']}")
        subprocess.run(command, check=True)
        candidates = sorted(Path(temporary).rglob("*.zip"))
        if len(candidates) != 1:
            names = ", ".join(str(path.name) for path in candidates)
            raise RuntimeError(f"expected one Kaggle archive, found {len(candidates)}: {names}")
        candidates[0].replace(destination)


def _safe_member_path(root: Path, member_name: str) -> Path:
    pure = PurePosixPath(member_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"refusing archive path outside extraction directory: {member_name!r}")
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"refusing archive path outside extraction directory: {member_name!r}") from error
    return target


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as archive_file:
        members = archive_file.infolist()
        for member in members:
            _safe_member_path(destination, member.filename)
            if member.is_dir():
                continue
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimeError(f"refusing symbolic link in archive: {member.filename!r}")
        for member in members:
            archive_file.extract(member, destination)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as archive_file:
        members = archive_file.getmembers()
        for member in members:
            _safe_member_path(destination, member.name)
            if not (member.isdir() or member.isreg()):
                raise RuntimeError(f"refusing non-file tar member: {member.name!r}")
        archive_file.extractall(destination, filter="data")


def extract_artifact(artifact: dict[str, Any]) -> None:
    if artifact["format"] == "file":
        return
    archive = _repo_path(artifact["path"])
    if not archive.is_file():
        raise RuntimeError(f"cannot extract missing artifact: {artifact['path']}")
    destination = _repo_path(artifact.get("extract_to", str(Path(artifact["path"]).parent)))
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {artifact['path']} -> {destination.relative_to(ROOT)}")
    if artifact["format"] == "zip":
        _extract_zip(archive, destination)
    elif artifact["format"] == "tar.gz":
        _extract_tar(archive, destination)
    else:
        raise RuntimeError(f"unsupported extraction format: {artifact['format']}")


def fetch_dataset(dataset: dict[str, Any], extract: bool, force: bool) -> bool:
    source = dataset["source"]
    success = True
    for artifact in dataset["artifacts"]:
        destination = _repo_path(artifact["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            if not verify_artifact(artifact):
                raise RuntimeError(
                    f"{artifact['path']} exists with the wrong checksum; use --force to replace it"
                )
        else:
            if destination.exists():
                destination.unlink()
            kind = source["kind"]
            if kind == "manual":
                raise RuntimeError(
                    f"download {source['page']} manually and save it as {artifact['path']}, then run verify"
                )
            if kind == "kaggle":
                _download_kaggle(dataset, destination)
            elif kind == "url":
                _download_url(artifact["url"], destination)
            else:
                raise RuntimeError(f"unsupported source kind: {kind}")
            if not verify_artifact(artifact):
                raise RuntimeError(f"checksum verification failed for {artifact['path']}")
        if extract:
            extract_artifact(artifact)
    return success


def command_list(registry: dict[str, Any]) -> int:
    for dataset in registry["datasets"]:
        source = dataset["source"]
        print(f"{dataset['id']}: {dataset['title']}")
        print(f"  source: {source['kind']} | {source['page']}")
        print(f"  license: {source.get('license', 'unspecified')}")
        for artifact in dataset["artifacts"]:
            print(f"  artifact: {artifact['path']} [{artifact['format']}]")
    return 0


def command_verify(registry: dict[str, Any], dataset_id: str | None) -> int:
    datasets = [_dataset(registry, dataset_id)] if dataset_id else registry["datasets"]
    return 0 if all(verify_artifact(artifact) for dataset in datasets for artifact in dataset["artifacts"]) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="list datasets, licenses, and artifacts")

    verify_parser = commands.add_parser("verify", help="verify available artifacts against SHA-256")
    verify_parser.add_argument("dataset", nargs="?", help="verify one dataset instead of all datasets")

    fetch_parser = commands.add_parser("fetch", help="download and optionally extract one dataset")
    fetch_parser.add_argument("dataset", help="dataset ID from datasets.json")
    fetch_parser.add_argument("--extract", action="store_true", help="extract archives after verification")
    fetch_parser.add_argument("--force", action="store_true", help="replace existing artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        registry = load_registry()
        if arguments.command == "list":
            return command_list(registry)
        if arguments.command == "verify":
            return command_verify(registry, arguments.dataset)
        if arguments.command == "fetch":
            fetch_dataset(_dataset(registry, arguments.dataset), arguments.extract, arguments.force)
            return 0
    except (RegistryError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
