#!/usr/bin/env python3
"""Create a fixed benchmark for the fast engine's declared support envelope."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import zxingcpp


FORMATS = (
    zxingcpp.BarcodeFormat.Code39,
    zxingcpp.BarcodeFormat.Code128,
    zxingcpp.BarcodeFormat.EAN13,
    zxingcpp.BarcodeFormat.QRCode,
    zxingcpp.BarcodeFormat.DataMatrix,
    zxingcpp.BarcodeFormat.PDF417,
    zxingcpp.BarcodeFormat.Aztec,
)
LINEAR = {
    zxingcpp.BarcodeFormat.Code39,
    zxingcpp.BarcodeFormat.Code128,
    zxingcpp.BarcodeFormat.EAN13,
}
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _ean13(generator: random.Random) -> str:
    digits = [generator.randrange(10) for _ in range(12)]
    checksum = (
        10 - (sum(digits[::2]) + 3 * sum(digits[1::2])) % 10
    ) % 10
    return "".join(str(value) for value in (*digits, checksum))


def _payload(
    generator: random.Random,
    barcode_format: zxingcpp.BarcodeFormat,
    page: int,
    index: int,
) -> str:
    if barcode_format == zxingcpp.BarcodeFormat.EAN13:
        return _ean13(generator)
    prefix = f"F{page:03d}{index:02d}"
    length = 14 if barcode_format in LINEAR else 28
    return prefix + "".join(generator.choice(ALPHABET) for _ in range(length - len(prefix)))


def _quad(
    center: tuple[float, float],
    width: int,
    height: int,
    angle: float,
) -> np.ndarray:
    radians = np.deg2rad(angle)
    rotation = np.asarray(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ],
        np.float32,
    )
    corners = np.asarray(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ],
        np.float32,
    )
    return corners @ rotation.T + np.asarray(center, np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=160)
    parser.add_argument("--seed", type=int, default=91_271)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    generator = random.Random(arguments.seed)
    records = []

    for page in range(1, arguments.pages + 1):
        width, height = (1800, 2400) if page % 2 else (2400, 1800)
        image = np.full((height, width), generator.randint(238, 255), np.uint8)
        for row in range(generator.randint(10, 28)):
            y = generator.randint(70, height - 70)
            x = generator.randint(60, max(61, width // 2))
            cv2.line(
                image,
                (x, y),
                (min(width - 60, x + generator.randint(160, 720)), y),
                generator.randint(75, 205),
                generator.randint(1, 3),
            )
        if generator.random() < 0.40:
            top = generator.randint(100, height // 2)
            left = generator.randint(100, width // 2)
            for offset in range(0, generator.randint(220, 650), 45):
                cv2.line(image, (left, top + offset), (width - 100, top + offset), 195, 2)

        objects = []
        symbol_count = 0 if page % 8 == 0 else generator.randint(1, 3)
        occupied: list[tuple[int, int, int, int]] = []
        for symbol_index in range(1, symbol_count + 1):
            barcode_format = generator.choice(FORMATS)
            payload = _payload(generator, barcode_format, page, symbol_index)
            try:
                symbol = np.asarray(
                    zxingcpp.create_barcode(payload, barcode_format).to_image(
                        scale=5,
                        add_hrt=False,
                        add_quiet_zones=True,
                    ),
                    np.uint8,
                )
            except Exception:
                continue
            desired_long = generator.randint(360, 760)
            scale = desired_long / max(symbol.shape)
            symbol = cv2.resize(
                symbol,
                (
                    max(24, int(round(symbol.shape[1] * scale))),
                    max(24, int(round(symbol.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_NEAREST,
            )
            angle = generator.choice((0, 0, 0, 90, -90, 12, -15, 28))
            symbol_height, symbol_width = symbol.shape
            placed = False
            for _ in range(100):
                center = (
                    generator.randint(desired_long // 2 + 80, width - desired_long // 2 - 80),
                    generator.randint(desired_long // 2 + 80, height - desired_long // 2 - 80),
                )
                candidate_quad = _quad(center, symbol_width, symbol_height, angle)
                x, y, box_width, box_height = cv2.boundingRect(candidate_quad)
                candidate = (x - 40, y - 40, x + box_width + 40, y + box_height + 40)
                if not any(
                    candidate[0] < prior[2]
                    and candidate[2] > prior[0]
                    and candidate[1] < prior[3]
                    and candidate[3] > prior[1]
                    for prior in occupied
                ):
                    placed = True
                    occupied.append(candidate)
                    break
            if not placed:
                continue
            source = np.asarray(
                [
                    [0, 0],
                    [symbol_width - 1, 0],
                    [symbol_width - 1, symbol_height - 1],
                    [0, symbol_height - 1],
                ],
                np.float32,
            )
            transform = cv2.getPerspectiveTransform(source, candidate_quad)
            warped = cv2.warpPerspective(
                symbol,
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            coverage = cv2.warpPerspective(
                np.full_like(symbol, 255),
                transform,
                (width, height),
                flags=cv2.INTER_NEAREST,
            )
            image[coverage > 0] = warped[coverage > 0]
            objects.append(
                {
                    "payload": payload,
                    "format": str(barcode_format),
                    "kind": "linear" if barcode_format in LINEAR else "2d",
                    "polygon": candidate_quad.tolist(),
                    "rotation_deg": angle,
                }
            )

        path = images_dir / f"page-{page:04d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append(
            {
                "id": path.name,
                "path": str(path),
                "width": width,
                "height": height,
                "objects": objects,
            }
        )

    manifest = {
        "name": "fast-engine-easy-documents-v1",
        "description": (
            "Large, clear, dark-on-light barcodes on document-like pages; "
            "fixed before fast-engine selection."
        ),
        "seed": arguments.seed,
        "images": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
