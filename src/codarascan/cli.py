# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for CodaraScan image, PDF, and worker operations."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from . import __version__
from .errors import (
    CodaraScanError,
    ConfigurationError,
    DocumentRenderError,
    InternalProcessingError,
    InvalidImageError,
    PageProcessingError,
)
from .models import DocumentResult, ImageResult
from .scanner import Scanner
from .serialization import to_json

EXIT_SUCCESS = 0
EXIT_NO_SYMBOLS = 1
EXIT_INVALID = 2
EXIT_PROCESSING = 3
EXIT_PARTIAL = 4


def _formats(value: str) -> list[str]:
    formats = [item.strip() for item in value.split(",") if item.strip()]
    if not formats:
        raise argparse.ArgumentTypeError("formats must not be empty")
    return formats


def _roi(value: str) -> tuple[float, float, float, float]:
    try:
        parts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must contain four values")
    return parts


def _pages(value: str) -> list[int]:
    output: list[int] = []
    try:
        for raw in value.split(","):
            part = raw.strip()
            if not part:
                continue
            if "-" in part:
                first, last = (int(item.strip()) for item in part.split("-", 1))
                if last < first:
                    raise ValueError
                output.extend(range(first, last + 1))
            else:
                output.append(int(part))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "pages must be one-based numbers or ascending ranges"
        ) from exc
    if not output or min(output) < 1:
        raise argparse.ArgumentTypeError("pages must contain positive page numbers")
    return output


def _workers(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be positive or 'auto'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("workers must be positive or 'auto'")
    return parsed


def _request_workers(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return parsed


def _add_scanner_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode", choices=("fast", "robust", "panorama"), default="fast"
    )
    parser.add_argument("--symbols", choices=("linear", "2d", "all"), default="all")
    parser.add_argument("--formats", type=_formats)
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--roi", type=_roi)
    parser.add_argument("--diagnostics", action="store_true")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true")
    output.add_argument("--ndjson", action="store_true")
    parser.add_argument(
        "--debug-output",
        type=Path,
        help="write a sensitivity-marked result manifest to a unique directory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codarascan")
    parser.add_argument(
        "--version",
        action="version",
        version=f"codarascan {__version__} (Apache-2.0)",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, metavar="{image,document}"
    )
    image = subparsers.add_parser("image", help="scan one image")
    image.add_argument("input", type=Path)
    _add_scanner_options(image)
    document = subparsers.add_parser("document", help="scan a PDF document")
    document.add_argument("input", type=Path)
    document.add_argument("--pages", type=_pages)
    document.add_argument("--workers", type=_workers, default=1)
    document.add_argument("--on-error", choices=("raise", "collect"), default="raise")
    _add_scanner_options(document)
    worker = subparsers.add_parser("_worker")
    worker.add_argument(
        "--workers", type=_request_workers, default=1,
        help="number of concurrent requests (default: 1)",
    )
    return parser


def _scanner(arguments: argparse.Namespace) -> Scanner:
    return Scanner(
        mode=arguments.mode,
        symbols=arguments.symbols,
        formats=arguments.formats,
        decode=not arguments.no_decode,
    )


def _human_image(result: ImageResult) -> str:
    lines = [
        f"{len(result.symbols)} symbol(s) in {result.elapsed_ms:.1f} ms "
        f"using {result.metadata.engine} ({result.metadata.backend})"
    ]
    for index, symbol in enumerate(result.symbols, start=1):
        payload = getattr(symbol, "text", "<geometry only>")
        format_name = getattr(symbol, "format", symbol.kind.value)
        lines.append(f"{index}. {symbol.status.value} {format_name}: {payload}")
    return "\n".join(lines)


def _human_document(result: DocumentResult) -> str:
    symbol_count = sum(len(page.image.symbols) for page in result.pages)
    return (
        f"{len(result.pages)} page(s), {symbol_count} symbol(s), "
        f"{len(result.errors)} error(s) in {result.elapsed_ms:.1f} ms"
    )


def _debug_result(result: ImageResult | DocumentResult, root: Path, diagnostics: bool) -> Path:
    destination = root.expanduser() / f"codarascan-{uuid.uuid4().hex}"
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    warning_path = destination / "SENSITIVE_DATA_WARNING.txt"
    warning_path.write_text(
        "This directory may contain sensitive decoded payloads and diagnostics.\n",
        encoding="utf-8",
    )
    warning_path.chmod(0o600)
    result_path = destination / "result.json"
    result_path.write_text(
        to_json(result, include_diagnostics=diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    result_path.chmod(0o600)
    return destination


def _has_symbols(result: ImageResult | DocumentResult) -> bool:
    if isinstance(result, ImageResult):
        return bool(result.symbols)
    return any(page.image.symbols for page in result.pages)


def _document_ndjson(arguments: argparse.Namespace, scanner: Scanner) -> int:
    pages = []
    with scanner.iter_document(
        arguments.input,
        pages=arguments.pages,
        workers=arguments.workers,
        on_error=arguments.on_error,
        roi=arguments.roi,
        diagnostics=arguments.diagnostics,
    ) as stream:
        for page in stream:
            pages.append(page)
            print(
                json.dumps(
                    page.to_dict(include_diagnostics=arguments.diagnostics),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        summary = {
            "type": "document_summary",
            "complete": stream.complete,
            "page_count": len(pages),
            "errors": [error.to_dict() for error in stream.errors],
            "elapsed_ms": stream.elapsed_ms,
            "metadata": stream.metadata.to_dict(),
        }
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if arguments.debug_output is not None:
            result = DocumentResult(
                tuple(pages),
                stream.errors,
                stream.complete,
                stream.elapsed_ms,
                stream.metadata,
            )
            destination = _debug_result(result, arguments.debug_output, arguments.diagnostics)
            print(f"debug output: {destination}", file=sys.stderr)
        if stream.errors:
            return EXIT_PARTIAL
        return EXIT_SUCCESS if any(page.image.symbols for page in pages) else EXIT_NO_SYMBOLS


def run(arguments: argparse.Namespace) -> int:
    if arguments.command == "_worker":
        from .worker import main as worker_main

        return worker_main(workers=arguments.workers)
    scanner = _scanner(arguments)
    if arguments.command == "image":
        result: ImageResult | DocumentResult = scanner.scan_image(
            arguments.input,
            roi=arguments.roi,
            diagnostics=arguments.diagnostics,
        )
    elif arguments.ndjson:
        return _document_ndjson(arguments, scanner)
    else:
        result = scanner.scan_document(
            arguments.input,
            pages=arguments.pages,
            workers=arguments.workers,
            on_error=arguments.on_error,
            roi=arguments.roi,
            diagnostics=arguments.diagnostics,
        )
    if arguments.json or arguments.ndjson:
        print(to_json(result, include_diagnostics=arguments.diagnostics))
    else:
        print(_human_image(result) if isinstance(result, ImageResult) else _human_document(result))
    if arguments.debug_output is not None:
        destination = _debug_result(result, arguments.debug_output, arguments.diagnostics)
        print(f"debug output: {destination}", file=sys.stderr)
    if isinstance(result, DocumentResult) and not result.complete:
        return EXIT_PARTIAL
    return EXIT_SUCCESS if _has_symbols(result) else EXIT_NO_SYMBOLS


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except (ConfigurationError, InvalidImageError) as exc:
        print(f"codarascan: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except (DocumentRenderError, PageProcessingError, InternalProcessingError) as exc:
        print(f"codarascan: {exc}", file=sys.stderr)
        return EXIT_PROCESSING
    except CodaraScanError as exc:
        print(f"codarascan: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXIT_INVALID",
    "EXIT_NO_SYMBOLS",
    "EXIT_PARTIAL",
    "EXIT_PROCESSING",
    "EXIT_SUCCESS",
    "build_parser",
    "main",
    "run",
]
