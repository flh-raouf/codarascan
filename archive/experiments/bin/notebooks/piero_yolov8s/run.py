#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_PDF = REPO_ROOT / "notebooks" / "data" / "pdfs" / "private evaluation document.pdf"
MODEL_REPO = "Piero2411/YOLOV8s-Barcode-Detection"
MODEL_FILE = "YOLOV8s_Barcode_Detection.pt"


@dataclass
class Detection:
    id: str
    class_id: int
    class_name: str
    confidence: float
    rect: dict[str, int]
    crop: str | None = None


def parse_pages(value: str) -> list[int]:
    pages: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = [int(part) for part in chunk.split("-", 1)]
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid page range: {chunk}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(chunk))
    if not pages or min(pages) < 1:
        raise argparse.ArgumentTypeError("pages must be positive numbers")
    return sorted(pages)


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def require_runtime() -> None:
    missing = []
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("pillow")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        missing.append("ultralytics")

    if missing:
        req = ROOT / "requirements.txt"
        raise SystemExit(
            "Missing Python packages: "
            + ", ".join(missing)
            + f"\nInstall them with: python3 -m pip install -r {req}"
        )
    if not shutil.which("pdftoppm"):
        raise SystemExit("Missing pdftoppm. Install Poppler first, for example: brew install poppler")


def download_model(model_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=model_dir,
        )
    )


def render_page(pdf: Path, page: int, pages_dir: Path, dpi: int, force: bool) -> Path:
    pages_dir.mkdir(parents=True, exist_ok=True)
    image = pages_dir / f"page-{page:03d}.png"
    if image.exists() and not force:
        return image

    prefix = pages_dir / f"page-{page:03d}"
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(page),
            "-l",
            str(page),
            "-r",
            str(dpi),
            "-singlefile",
            "-png",
            str(pdf),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not image.exists():
        raise RuntimeError(f"pdftoppm did not create {image}")
    return image


def draw_debug(image_path: Path, detections: list[Detection], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    colors = {
        "barcode": "#22c55e",
        "qrcode": "#3b82f6",
    }
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for det in detections:
        rect = det.rect
        color = colors.get(det.class_name.lower(), "#f97316")
        xy = [rect["x1"], rect["y1"], rect["x2"], rect["y2"]]
        draw.rectangle(xy, outline=color, width=4)
        label = f"{det.class_name} {det.confidence:.2f}"
        box = draw.textbbox((xy[0], xy[1]), label, font=font)
        label_bg = [box[0] - 2, box[1] - 2, box[2] + 4, box[3] + 3]
        draw.rectangle(label_bg, fill=color)
        draw.text((xy[0] + 1, xy[1]), label, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_crop(image_path: Path, det: Detection, out_path: Path, padding: int) -> None:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    rect = det.rect
    crop_box = (
        max(0, rect["x1"] - padding),
        max(0, rect["y1"] - padding),
        min(width, rect["x2"] + padding),
        min(height, rect["y2"] + padding),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(crop_box).save(out_path)


def detect_page(model, image_path: Path, args: argparse.Namespace, out_dir: Path) -> dict:
    from PIL import Image

    result = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        verbose=False,
    )[0]

    width, height = Image.open(image_path).size
    names = result.names
    detections: list[Detection] = []
    page = int(image_path.stem.split("-")[-1])

    for index, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        class_name = str(names.get(class_id, class_id))
        det_id = f"page-{page:03d}-box-{index:03d}"
        det = Detection(
            id=det_id,
            class_id=class_id,
            class_name=class_name,
            confidence=round(float(box.conf[0]), 6),
            rect={
                "x1": max(0, min(width, x1)),
                "y1": max(0, min(height, y1)),
                "x2": max(0, min(width, x2)),
                "y2": max(0, min(height, y2)),
            },
        )
        detections.append(det)

    detections.sort(key=lambda item: (item.rect["y1"], item.rect["x1"]))
    for index, det in enumerate(detections, start=1):
        det.id = f"page-{page:03d}-box-{index:03d}"
        if args.save_crops:
            crop_path = out_dir / "crops" / f"{det.id}-{det.class_name}.png"
            save_crop(image_path, det, crop_path, args.crop_padding)
            det.crop = rel(crop_path, out_dir)

    debug_path = out_dir / "debug" / f"page-{page:03d}-debug.png"
    draw_debug(image_path, detections, debug_path)

    return {
        "page": page,
        "image": rel(image_path, out_dir),
        "debug": rel(debug_path, out_dir),
        "width": width,
        "height": height,
        "detections": [asdict(det) for det in detections],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Piero YOLOv8s barcode detection on PDF pages.")
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out", type=Path, default=ROOT / "output")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--pages", type=parse_pages, default=parse_pages("16,20,21"))
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-crops", action="store_true", default=True)
    parser.add_argument("--no-save-crops", dest="save_crops", action="store_false")
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--force-render", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf = args.pdf.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()

    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")
    require_runtime()

    from ultralytics import YOLO

    start = perf_counter()
    model_path = download_model(args.model_dir.expanduser().resolve())
    model = YOLO(str(model_path))

    pages = []
    for page in args.pages:
        image = render_page(pdf, page, out_dir / "pages", args.dpi, args.force_render)
        pages.append(detect_page(model, image, args, out_dir))

    payload = {
        "model": MODEL_REPO,
        "weights": rel(model_path, REPO_ROOT),
        "model_names": {str(key): value for key, value in model.names.items()},
        "pdf": rel(pdf, REPO_ROOT),
        "dpi": args.dpi,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "elapsed_seconds": round(perf_counter() - start, 3),
        "pages": pages,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = out_dir / "detections.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
