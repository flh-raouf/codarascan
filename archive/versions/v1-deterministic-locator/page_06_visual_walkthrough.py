# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (barcode locator)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deterministic barcode locator - complete visual walkthrough
#
# ## Page 6 of `private-evaluation-document.pdf`
#
# This notebook opens the implementation in `barcode_locator.py` and exposes the
# complete page-6 journey:
#
# 1. render the PDF page at its useful native resolution;
# 2. collect whole-page ZXing results;
# 3. run **every actual OpenCV detector pass** used by the implementation;
# 4. visualize gradients and directional coherence;
# 5. show the optional structure-tensor route, including its internal maps;
# 6. merge duplicate proposals;
# 7. perspective-rectify every candidate;
# 8. calculate every structural metric and every acceptance gate;
# 9. attempt local decoding;
# 10. produce final boxes, crops, tables and JSON-like output.
#
# The notebook saves high-resolution figures under
# `output/notebook-page-06/figures/`, so they can be reused in presentation slides.
#
# > **Accuracy note:** OpenCV's Python API returns its final quadrilaterals but
# > does not expose private intermediate buffers. This notebook therefore keeps
# > two things separate: **exact actual detector outputs** and a clearly labeled
# > **educational reconstruction** of the gradient/coherence idea described by
# > OpenCV. The structure-tensor maps, structural verifier and final decisions
# > are produced directly from the equations and code used by our implementation.

# %% [markdown]
# ## 0. Configuration and imports
#
# Run the notebook from its own folder. The path-discovery code also works when
# Jupyter starts in the repository root.

# %%
from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (14, 8),
        "figure.dpi": 110,
        "savefig.dpi": 180,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)

PAGE_NUMBER = 6
DPI = 200
ANGLE_STEP = 15
MINIMUM_SCORE = 0.52
MINIMUM_LINEAR_LENGTH = 90.0
RUN_OPTIONAL_TENSOR = True


def find_project_dir() -> Path:
    here = Path.cwd().resolve()
    candidates = [
        here,
        here / "archive" / "versions" / "v1-deterministic-locator",
        here.parent / "archive" / "versions" / "v1-deterministic-locator",
    ]
    for candidate in candidates:
        if (candidate / "barcode_locator.py").exists():
            return candidate
    raise FileNotFoundError("Could not find archive/versions/v1-deterministic-locator/barcode_locator.py")


PROJECT_DIR = find_project_dir()
REPOSITORY_DIR = PROJECT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import barcode_locator as bl  # noqa: E402

PDF_CANDIDATES = [
    REPOSITORY_DIR / "archive" / "experiments" / "notebooks" / "data" / "pdfs" / "private-evaluation-document.pdf",
    REPOSITORY_DIR / "archive" / "experiments" / "bin" / "data" / "pdfs" / "private-evaluation-document.pdf",
    REPOSITORY_DIR / "private-evaluation-document.pdf",
]
PDF_PATH = next((path for path in PDF_CANDIDATES if path.exists()), None)

OUTPUT_DIR = PROJECT_DIR / "output" / "notebook-page-06"
FIGURE_DIR = OUTPUT_DIR / "figures"
RENDER_DIR = OUTPUT_DIR / "rendered"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project: {PROJECT_DIR}")
print(f"Locator: {PROJECT_DIR / 'barcode_locator.py'}")
print(f"PDF:     {PDF_PATH}")
print(f"Output:  {OUTPUT_DIR}")
print(f"OpenCV:  {cv2.__version__}")
print(f"ZXing:   {'available' if bl.zxingcpp is not None else 'NOT available'}")

# %% [markdown]
# ### Reusable display helpers
#
# These helpers only draw figures and tables. They do not alter the detector.

# %%
COLORS = [
    (255, 70, 70),
    (70, 190, 70),
    (70, 120, 255),
    (230, 170, 30),
    (180, 80, 220),
    (20, 190, 190),
]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def normalize_for_display(array: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    high = float(np.percentile(np.abs(values), percentile))
    if high <= bl.EPS:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(np.abs(values) / high * 255.0, 0, 255).astype(np.uint8)


def save_and_show(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.show()


def show_image(
    image: np.ndarray,
    title: str,
    filename: str | None = None,
    figsize: tuple[float, float] = (13, 9),
    cmap: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(bgr_to_rgb(image), cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    if filename:
        save_and_show(fig, filename)
    else:
        plt.show()


def draw_detections(
    image: np.ndarray,
    detections: Sequence[bl.Detection],
    color_by_source: bool = False,
    thickness: int = 4,
) -> np.ndarray:
    output = image.copy()
    for index, detection in enumerate(detections, start=1):
        if color_by_source:
            source_text = " ".join(sorted(detection.sources))
            if "zxing" in source_text:
                color = (40, 210, 40)
            elif "tensor" in source_text:
                color = (220, 90, 220)
            else:
                color = (0, 170, 255)
        else:
            color = COLORS[(index - 1) % len(COLORS)][::-1]
        quad = np.round(detection.quad).astype(np.int32)
        cv2.polylines(output, [quad], True, color, thickness, cv2.LINE_AA)
        anchor = np.min(quad, axis=0)
        label = str(index)
        if detection.decoded_text:
            label += f" {detection.decoded_text[:22]}"
        cv2.putText(
            output,
            label,
            (int(anchor[0]), max(28, int(anchor[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def detections_dataframe(detections: Sequence[bl.Detection], prefix: str = "C") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, detection in enumerate(detections, start=1):
        long_side, short_side = detection.long_short()
        rows.append(
            {
                "id": f"{prefix}{index}",
                "sources": ", ".join(sorted(detection.sources)),
                "proposal_score": detection.proposal_score,
                "structural_score": detection.structural_score,
                "decoded_text": detection.decoded_text,
                "format": detection.barcode_format,
                "center_x": round(float(detection.center()[0]), 1),
                "center_y": round(float(detection.center()[1]), 1),
                "angle_deg": round(detection.angle_degrees(), 1),
                "long_px": round(long_side, 1),
                "short_px": round(short_side, 1),
            }
        )
    return pd.DataFrame(rows)


def crop_around_quad(image: np.ndarray, quad: np.ndarray, padding: int = 30) -> np.ndarray:
    x_min, y_min = np.floor(np.min(quad, axis=0)).astype(int) - padding
    x_max, y_max = np.ceil(np.max(quad, axis=0)).astype(int) + padding
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)
    return image[y_min:y_max, x_min:x_max]


def source_family(detection: bl.Detection) -> str:
    text = " ".join(detection.sources)
    if "zxing" in text:
        return "ZXing"
    if "tensor" in text:
        return "Tensor"
    return "OpenCV"


def explain_visual(
    figure_name: str,
    how_to_read: Sequence[str],
    page_6_observation: str,
    why_it_matters: str,
) -> None:
    """Display a consistent beginner-friendly guide immediately before a visual."""
    instructions = "\n".join(f"{index}. {item}" for index, item in enumerate(how_to_read, start=1))
    display(
        Markdown(
            f"""
### How to read {figure_name}

{instructions}

**What to notice on page 6:** {page_6_observation}

**Why this visual matters:** {why_it_matters}
"""
        )
    )

# %% [markdown]
# ## Small visual vocabulary used throughout the notebook
#
# | Word | Simple meaning | How it normally appears |
# |---|---|---|
# | **Pixel** | One tiny image square containing a brightness/color value | A position in the page image |
# | **Gradient** | How quickly brightness changes between neighboring pixels | Bright values mean a strong edge |
# | **Heatmap / response map** | A numeric score at every pixel | Dark = low score, bright = high score |
# | **Binary mask** | A yes/no image created after thresholding | Black = rejected, white = selected |
# | **Proposal / candidate** | A region that might contain a barcode | A colored rotated outline |
# | **Threshold** | The boundary between keeping and rejecting a value | Often shown as a red dashed line |
# | **Rectification** | Straightening a rotated quadrilateral into a rectangle | Rotated page label becomes a horizontal crop |
# | **Transition** | A black-to-white or white-to-black change | A vertical marker in a 1-D signal plot |
# | **Gate** | One required true/false test | A Boolean column in a table |
#
# ### Color warning
#
# Colors are local to each figure. For example, magenta means **tensor proposal**
# in the tensor figure, while it means **decoded 2-D matrix barcode** in the final
# production overlay. Always read the title and the guide directly above a figure.

# %% [markdown]
# # Step 1 - Render page 6
#
# The dossier page is rendered at **200 DPI** because this PDF already contains
# an approximately 200-ppi raster. A much larger DPI would mostly interpolate
# existing pixels rather than recover new bar detail.

# %%
if PDF_PATH is None:
    fallback = PROJECT_DIR / "output" / "dossier-rerun" / "rendered-pages" / "pages" / "page-06.png"
    if not fallback.exists():
        raise FileNotFoundError("private-evaluation-document.pdf and the page-6 fallback image are both missing")
    PAGE_PATH = fallback
    render_command = "Used the existing rendered page because the source PDF was not found."
else:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("Poppler/pdftoppm is required to render the PDF")
    prefix = RENDER_DIR / f"page-{PAGE_NUMBER:02d}"
    PAGE_PATH = prefix.with_suffix(".png")
    command = [
        "pdftoppm",
        "-f",
        str(PAGE_NUMBER),
        "-l",
        str(PAGE_NUMBER),
        "-r",
        str(DPI),
        "-singlefile",
        "-png",
        str(PDF_PATH),
        str(prefix),
    ]
    subprocess.run(command, check=True)
    render_command = " ".join(command)

image = cv2.imread(str(PAGE_PATH), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError(f"OpenCV could not read {PAGE_PATH}")
gray = bl.as_gray(image)

print(render_command)
print(f"Rendered page: {PAGE_PATH}")
print(f"Shape: {image.shape[1]} x {image.shape[0]} pixels, {image.shape[2]} color channels")
print(f"Memory: {image.nbytes / 1024**2:.1f} MiB")

explain_visual(
    "Figure 01 - the rendered input page",
    [
        "Read it like a normal scanned document; no detection has happened yet.",
        "Look for compact groups of repeated black vertical stripes. Those are the physical barcode clues.",
        "Ignore the large pale background, ordinary text, and table grid for now; the detector must later reject them.",
    ],
    "There are three real 1-D barcodes: a small one at the upper right of the form, a long VIN barcode directly below it, and a slightly tilted component barcode lower on the page.",
    "This is the baseline. Every later map and box must be explainable from pixels already visible here.",
)

show_image(
    image,
    f"Step 1 - private evaluation document, page {PAGE_NUMBER}, rendered at {DPI} DPI",
    "01-rendered-page-06.png",
    figsize=(11, 15),
)

# %% [markdown]
# ### What does the computer receive?
#
# The detector receives arrays of pixel intensities. The grayscale image removes
# color while keeping brightness. The histogram shows how many pixels have each
# intensity from black (0) to white (255).

# %%
explain_visual(
    "Figure 02 - grayscale image and intensity histogram",
    [
        "Left: black is intensity 0, white is 255, and gray is between them. The algorithms mainly work from these brightness values.",
        "Right: the horizontal axis is brightness; the vertical axis is how many page pixels have that brightness.",
        "A high bar near 255 means that most of the scan is white paper. The much smaller dark part contains text, tables, barcode bars, and noise together.",
    ],
    "The histogram cannot separate barcode bars from letters or table lines; all of them contribute dark pixels.",
    "This shows why simple global thresholding is insufficient: brightness alone does not tell us what kind of dark mark we found.",
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Grayscale page used by the visual detectors")
axes[0].axis("off")
axes[1].hist(gray.ravel(), bins=256, range=(0, 256), color="#374151")
axes[1].set_title("Page grayscale histogram")
axes[1].set_xlabel("Intensity: 0 = black, 255 = white")
axes[1].set_ylabel("Number of pixels")
save_and_show(fig, "02-grayscale-and-histogram.png")

# %% [markdown]
# # Step 2 - Generate candidate regions
#
# A candidate is only a hypothesis: "there may be a barcode here." The standard
# implementation combines:
#
# - ZXing whole-page reads - very strong when decoding succeeds;
# - OpenCV directional-coherence proposals - the main visual proposal source;
# - optional structure-tensor proposals - a slower, independent recall route.
#
# Candidate generation is deliberately permissive. Strict verification happens
# later, after each region has been straightened.

# %% [markdown]
# ## Step 2A - ZXing whole-page proposals
#
# ZXing simultaneously searches and decodes. Any successful result supplies four
# corner points, a symbology and decoded text. Failure does **not** prove that no
# barcode is present.

# %%
zxing_candidates = bl.decoder_proposals(image)
print(f"ZXing found {len(zxing_candidates)} whole-page result(s).")
display(detections_dataframe(zxing_candidates, "Z"))

explain_visual(
    "Figure 03 - whole-page ZXing results",
    [
        "Each green quadrilateral is a barcode that ZXing both located and decoded successfully.",
        "The number identifies the result in the table above; the nearby text is the decoded value.",
        "A region without a green box is not automatically non-barcode: it may simply be too difficult for the whole-page decoder.",
    ],
    "ZXing decodes the two barcodes near the top. The tilted lower barcode is visible on the page but is not returned in this whole-page pass.",
    "The missing lower code proves that decoder success cannot be the only localization rule.",
)

show_image(
    draw_detections(image, zxing_candidates, color_by_source=True),
    "Step 2A - Exact ZXing whole-page results (green)",
    "03-zxing-whole-page-results.png",
    figsize=(11, 15),
)

# %% [markdown]
# ## Step 2B - OpenCV's actual directional-coherence detector
#
# The production implementation calls `cv2.barcode.BarcodeDetector.detectMulti`.
# It does not merely run our own morphology. OpenCV's documented classical idea
# is:
#
# ```text
# Scharr gradients -> local patches -> directional coherence
#      -> reject isolated patches -> connect similar orientations
#      -> fit rotated rectangles
# ```
#
# The next cells execute **every real detector call** made by
# `detector_proposals(gray, angle_step=15)`.

# %% [markdown]
# ### Detector scale profiles
#
# These values do not resize the page. They change OpenCV's internal box-filter
# or analysis-neighborhood sizes. The values are relative to the page's smaller
# dimension.

# %%
minimum_dimension = min(gray.shape)
scale_rows = []
for profile_name, scales in bl.SCALE_PROFILES:
    for scale in scales:
        scale_rows.append(
            {
                "profile": profile_name,
                "relative_scale": scale,
                "approx_filter_size_px": round(scale * minimum_dimension, 2),
            }
        )
scale_table = pd.DataFrame(scale_rows)
display(
    Markdown(
        """
**How to read the scale table:** each row is one internal analysis size. `relative_scale`
is multiplied by the page's smaller dimension to obtain the approximate pixel size
shown in the last column. These are neighborhood/filter sizes; they are **not new
resolutions of the complete page**.
"""
    )
)
display(scale_table)

explain_visual(
    "Figure 04 - detector scale profiles",
    [
        "Each dot is one analysis-neighborhood size passed to OpenCV.",
        "Moving right/up means using a larger neighborhood. Both axes use logarithmic spacing, so equal visual gaps represent multiplication rather than addition.",
        "Colors group the sizes into our `tiny`, `document`, and `wide` profiles. Overlap between colors is intentional.",
    ],
    "The small header code, long VIN code, and lower component code have different physical dimensions, so one neighborhood size would be fragile.",
    "Trying several neighborhood sizes makes the same unchanged page searchable for small and large barcode structures.",
)

fig, ax = plt.subplots(figsize=(13, 5))
for profile_name, group in scale_table.groupby("profile"):
    ax.scatter(group["relative_scale"], group["approx_filter_size_px"], s=70, label=profile_name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Scale passed to OpenCV")
ax.set_ylabel("Approximate scale x min(page width, page height), in pixels")
ax.set_title("Same page, several analysis-neighborhood sizes")
ax.legend()
save_and_show(fig, "04-detector-scale-profiles.png")

# %% [markdown]
# ### Execute every actual OpenCV pass
#
# Native page:
#
# - 3 profiles x 2 gradient thresholds = 6 calls.
#
# Coarse rotations:
#
# - 5 angles x 1 document profile x 2 thresholds = 10 calls.
#
# Total: **16 real OpenCV detector calls** for page 6 with `--angle-step 15`.

# %%
def run_actual_opencv_pass(
    input_gray: np.ndarray,
    profile_name: str,
    scales: Sequence[float],
    threshold: float,
    source_name: str,
    inverse_transform: np.ndarray | None = None,
) -> list[bl.Detection]:
    detector = bl.make_detector(scales, input_gray.shape, threshold)
    try:
        found, points = detector.detectMulti(input_gray)
    except cv2.error:
        return []
    if not found or points is None:
        return []
    output: list[bl.Detection] = []
    for quad in np.asarray(points, dtype=np.float32):
        if inverse_transform is not None:
            quad = cv2.transform(quad.reshape(1, -1, 2), inverse_transform)[0]
        detection = bl.Detection(
            quad=quad,
            sources={f"opencv:{source_name}:{profile_name}:g{int(threshold)}"},
            proposal_score=0.48,
        )
        long_side, short_side = detection.long_short()
        if long_side < 18 or short_side < 5 or long_side / (short_side + bl.EPS) < 1.15:
            continue
        output.append(detection)
    return output


opencv_passes: list[dict[str, Any]] = []

for profile_name, scales in bl.SCALE_PROFILES:
    for threshold in (48.0, 64.0):
        detections = run_actual_opencv_pass(
            gray, profile_name, scales, threshold, "native"
        )
        opencv_passes.append(
            {
                "angle": 0,
                "profile": profile_name,
                "threshold": int(threshold),
                "detections": detections,
            }
        )

document_scales = bl.SCALE_PROFILES[1][1]
rotated_pages: dict[int, tuple[np.ndarray, np.ndarray]] = {}
for angle in range(ANGLE_STEP, 90, ANGLE_STEP):
    rotated, inverse = bl.rotate_bound(gray, angle)
    rotated_pages[angle] = (rotated, inverse)
    for threshold in (48.0, 64.0):
        detections = run_actual_opencv_pass(
            rotated,
            "deskew",
            document_scales,
            threshold,
            f"deskew{angle}",
            inverse,
        )
        opencv_passes.append(
            {
                "angle": angle,
                "profile": "deskew",
                "threshold": int(threshold),
                "detections": detections,
            }
        )

opencv_pass_table = pd.DataFrame(
    [
        {
            "pass": index,
            "page_rotation_deg": item["angle"],
            "profile": item["profile"],
            "gradient_threshold": item["threshold"],
            "number_of_proposals": len(item["detections"]),
            "angles_returned": [round(det.angle_degrees(), 1) for det in item["detections"]],
        }
        for index, item in enumerate(opencv_passes, start=1)
    ]
)
display(
    Markdown(
        """
**How to read the pass table:** one row equals one real call to OpenCV.
`page_rotation_deg` says which temporary canvas was inspected; `profile` says which
scale group was used; `gradient_threshold` is the edge/coherence setting; and
`number_of_proposals` is how many quadrilaterals that single call returned.
`angles_returned` contains the estimated long-axis angles after mapping boxes back
to the original page.
"""
    )
)
display(opencv_pass_table)

# %% [markdown]
# ### Visual result of all 16 actual OpenCV calls
#
# A blank panel means that particular configuration returned no candidate. The
# coordinates displayed below have already been mapped back to the original
# unrotated page.

# %%
explain_visual(
    "Figure 05 - all 16 OpenCV detector calls",
    [
        "Read each panel independently. Its title gives temporary page rotation, scale profile, gradient threshold `g`, and proposal count.",
        "The first six panels are the native page: 3 profiles times 2 thresholds. The remaining ten panels are 5 rotations times 2 thresholds.",
        "Colored outlines are raw proposals from that one call. Their colors only separate boxes visually; they are not confidence levels.",
        "Compare positions across panels: boxes repeatedly appearing over the same physical stripe group are duplicate evidence for one barcode.",
    ],
    "Most passes repeatedly find the same three real barcode locations. Some tiny-profile passes return a fourth region, showing that proposal generation is deliberately permissive.",
    "This grid shows exactly where the 53 raw OpenCV proposals come from and why deduplication and later verification are necessary.",
)

fig, axes = plt.subplots(4, 4, figsize=(18, 23))
for axis, item in zip(axes.ravel(), opencv_passes):
    overlay = draw_detections(image, item["detections"])
    axis.imshow(bgr_to_rgb(overlay))
    axis.set_title(
        f"rotation={item['angle']} deg | {item['profile']} | g={item['threshold']}\n"
        f"{len(item['detections'])} proposal(s)"
    )
    axis.axis("off")
save_and_show(fig, "05-all-16-opencv-passes.png")

# %%
opencv_candidates = [
    detection
    for detector_pass in opencv_passes
    for detection in detector_pass["detections"]
]
print(f"OpenCV returned {len(opencv_candidates)} raw proposals across all 16 calls.")
display(detections_dataframe(opencv_candidates, "O"))

explain_visual(
    "Figure 06 - all raw OpenCV proposals on one page",
    [
        "Every outline from all 16 calls is drawn on the original page at once.",
        "Many outlines lie almost exactly on top of one another, so 53 proposals do not look like 53 separate objects.",
        "The small numbers identify raw objects only. They will not become final barcode numbers.",
    ],
    "The dense stacks of outlines concentrate around the three true barcodes. Overlap is evidence that several settings agree, not evidence of many different codes.",
    "This is the input to duplicate removal. It must be reduced to one candidate per physical barcode.",
)

show_image(
    draw_detections(image, opencv_candidates),
    "Step 2B - All raw OpenCV proposals before duplicate removal",
    "06-all-raw-opencv-proposals.png",
    figsize=(11, 15),
)

# %% [markdown]
# ### Educational reconstruction of OpenCV's hidden gradient idea
#
# The following maps are **not private OpenCV buffers**. The Python API cannot
# retrieve those. They are a transparent reconstruction of the same physical
# evidence described by OpenCV: strong edges that agree on one direction.
#
# We zoom into the largest OpenCV proposal because full-page gradient maps are
# difficult to read on a presentation slide.

# %%
if not opencv_candidates:
    raise RuntimeError("No OpenCV candidates were returned; the visual walkthrough needs at least one")

largest_opencv = max(opencv_candidates, key=lambda item: item.area())
zoom_gray = crop_around_quad(gray, largest_opencv.quad, padding=70)
blurred = cv2.GaussianBlur(zoom_gray, (0, 0), 0.8)
gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
magnitude = cv2.magnitude(gx, gy)
orientation = np.arctan2(gy, gx)

hsv = np.zeros((*zoom_gray.shape, 3), dtype=np.uint8)
hsv[..., 0] = np.mod((orientation + math.pi) * 180.0 / (2.0 * math.pi), 180).astype(np.uint8)
hsv[..., 1] = 255
hsv[..., 2] = normalize_for_display(magnitude)
orientation_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

explain_visual(
    "Figure 07 - construction of edge evidence",
    [
        "Start at the top left and read left-to-right, then continue on the second row.",
        "Original and blur: the blur removes tiny scanner noise while preserving the larger bar edges.",
        "`|Scharr Gx|` becomes bright where brightness changes while moving left/right. Vertical barcode edges therefore appear as many bright vertical lines.",
        "`|Scharr Gy|` becomes bright at horizontal edges. A clean vertical-bar field should have much stronger repeated Gx than Gy.",
        "Gradient magnitude ignores direction and shows total edge strength. The orientation panel uses hue for direction and brightness for strength; similar colors mean similar directions.",
    ],
    "The barcode region creates a dense comb of bright, similarly oriented vertical edges. Ordinary nearby marks are shorter or point in mixed directions.",
    "The detector is looking for organized edge direction, not simply black ink or a rectangular dark blob.",
)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
panels = [
    (zoom_gray, "Original grayscale crop", "gray"),
    (blurred, "Small Gaussian blur", "gray"),
    (normalize_for_display(gx), "|Scharr Gx|: left-right changes", "magma"),
    (normalize_for_display(gy), "|Scharr Gy|: up-down changes", "magma"),
    (normalize_for_display(magnitude), "Gradient magnitude", "inferno"),
    (orientation_rgb, "Gradient orientation (hue) + strength", None),
]
for axis, (panel, title, cmap) in zip(axes.ravel(), panels):
    axis.imshow(panel, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
save_and_show(fig, "07-gradient-construction.png")

# %% [markdown]
# ### Local directional coherence
#
# We use the local structure-matrix coherence equation as an interpretable proxy:
#
# ```text
# coherence = sqrt((Cxx-Cyy)^2 + 4*Cxy^2) / (Cxx+Cyy)
# ```
#
# - near 0: local edge directions disagree;
# - near 1: local edges strongly agree on one direction.
#
# Strong coherence alone is not enough: a table line is coherent too. We also
# require edge energy and later verify repeated black/white transitions.

# %%
proxy_window = 15
cxx = cv2.boxFilter(gx * gx, cv2.CV_32F, (proxy_window, proxy_window), normalize=True)
cyy = cv2.boxFilter(gy * gy, cv2.CV_32F, (proxy_window, proxy_window), normalize=True)
cxy = cv2.boxFilter(gx * gy, cv2.CV_32F, (proxy_window, proxy_window), normalize=True)
trace = cxx + cyy
coherence_proxy = np.sqrt((cxx - cyy) ** 2 + 4.0 * cxy**2) / (trace + bl.EPS)
energy_proxy = np.sqrt(trace)
energy_high = float(np.percentile(energy_proxy, 99.2))
response_proxy = coherence_proxy * np.minimum(energy_proxy / max(energy_high, bl.EPS), 1.0)
proxy_threshold = max(0.07, float(np.percentile(response_proxy, 99.45)))
proxy_mask = (response_proxy >= proxy_threshold) & (coherence_proxy >= 0.52)

explain_visual(
    "Figure 08 - local directional-coherence proxy",
    [
        "Coherence: dark/purple is weak directional agreement; bright/yellow is strong agreement. A value near 1 means nearby edges mostly share one direction.",
        "Energy: bright means strong edges. A pale but perfectly aligned region can still have low energy.",
        "Response multiplies directional agreement by normalized edge energy, so a region must have both properties.",
        "Threshold mask is a yes/no result: white pixels pass the response and coherence conditions; black pixels do not.",
    ],
    "The barcode's stripe band survives as a concentrated white response. Many weak background pixels disappear, although this mask is still only proposal evidence.",
    "This makes the phrase `directional coherence` visible. The figure is an educational proxy, not an internal OpenCV debug buffer.",
)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
proxy_panels = [
    (coherence_proxy, "Directional coherence, 0 to 1", "viridis"),
    (normalize_for_display(energy_proxy), "Local edge energy", "inferno"),
    (response_proxy, "coherence x normalized energy", "magma"),
    (proxy_mask, f"threshold mask (response >= {proxy_threshold:.3f})", "gray"),
]
for axis, (panel, title, cmap) in zip(axes, proxy_panels):
    axis.imshow(panel, cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
save_and_show(fig, "08-directional-coherence-proxy.png")

print(f"Proxy window: {proxy_window} x {proxy_window} pixels")
print(f"99.2th-percentile energy reference: {energy_high:.3f}")
print(f"Response threshold: {proxy_threshold:.4f}")
print(f"Selected pixels: {100.0 * float(np.mean(proxy_mask)):.3f}% of the zoomed crop")

# %% [markdown]
# # Step 3 - Optional complete-page rotations
#
# With `--angle-step 15`, this is enabled for every page. It is "optional"
# because `--angle-step 0` disables it.
#
# The implementation rotates without clipping the corners and retains an inverse
# affine matrix. Any detector quadrilateral returned on a rotated canvas is
# transformed back into original page coordinates.

# %%
explain_visual(
    "Figure 09 - full-canvas rotation attempts",
    [
        "Every panel contains the same page, temporarily rotated by the angle in its title.",
        "The canvas grows so no original page corner is cut off. The added empty area is expected and contains no new information.",
        "These panels show detector inputs, not final results. Detected points are later transformed back to the original unrotated coordinate system.",
    ],
    "The lower barcode and page grid change their pixel alignment at each angle. A weak barcode may be easier for OpenCV to group in one of these samplings.",
    "Rotation creates additional deterministic opportunities without forcing the user to know the barcode angle in advance.",
)

fig, axes = plt.subplots(1, len(rotated_pages), figsize=(20, 7))
for axis, (angle, (rotated, _inverse)) in zip(axes, rotated_pages.items()):
    axis.imshow(rotated, cmap="gray")
    axis.set_title(f"+{angle} deg\n{rotated.shape[1]} x {rotated.shape[0]}")
    axis.axis("off")
save_and_show(fig, "09-full-canvas-rotations.png")

rotation_summary = (
    opencv_pass_table.groupby("page_rotation_deg", as_index=False)["number_of_proposals"]
    .sum()
    .rename(columns={"number_of_proposals": "raw_proposals_across_profiles_and_thresholds"})
)
display(
    Markdown(
        """
**How to read the rotation summary:** each row adds the proposal counts from the
two thresholds used at that temporary rotation. It counts raw detector returns,
not unique physical barcodes; the same code can contribute once at every angle.
"""
    )
)
display(rotation_summary)

# %% [markdown]
# The extra rotations are a **recall safety net**, not the final angle estimate.
# OpenCV still returns an oriented quadrilateral, and the inverse transform maps
# that quadrilateral back to the native page.

# %% [markdown]
# # Step 2C - Optional independent structure-tensor proposals
#
# This route is related to directional coherence but implemented independently.
# It works per pixel, builds a local 2 x 2 structure tensor, pools evidence at
# three scales, separates 18 orientation bins, closes small gaps, extracts
# components and fits rotated rectangles.
#
# It is disabled in the normal command unless `--tensor-fallback` or
# `--high-recall` is requested. We run it here for education.

# %%
def thumbnail(array: np.ndarray, max_dimension: int = 900, nearest: bool = False) -> np.ndarray:
    height, width = array.shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    if scale >= 1.0:
        return array.copy()
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(array, None, fx=scale, fy=scale, interpolation=interpolation)


def trace_tensor_maps(input_gray: np.ndarray, max_dimension: int = 1900) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = input_gray.shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    work = (
        cv2.resize(input_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else input_gray
    )
    blur = cv2.GaussianBlur(work, (0, 0), 0.8)
    tensor_gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    tensor_gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    records: list[dict[str, Any]] = []

    for local_size, pool_size in ((7, 19), (13, 35), (21, 61)):
        cxx = cv2.boxFilter(tensor_gx * tensor_gx, cv2.CV_32F, (local_size, local_size), normalize=True)
        cyy = cv2.boxFilter(tensor_gy * tensor_gy, cv2.CV_32F, (local_size, local_size), normalize=True)
        cxy = cv2.boxFilter(tensor_gx * tensor_gy, cv2.CV_32F, (local_size, local_size), normalize=True)
        trace = cxx + cyy
        coherence = np.sqrt((cxx - cyy) ** 2 + 4.0 * cxy**2) / (trace + bl.EPS)
        energy = np.sqrt(trace)
        high = float(np.percentile(energy, 99.2))
        edge_evidence = coherence * np.minimum(energy / max(high, bl.EPS), 1.0)
        response = cv2.boxFilter(edge_evidence, cv2.CV_32F, (pool_size, pool_size), normalize=True)
        orientation = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
        threshold = max(0.07, float(np.percentile(response, 99.45)))
        base_mask = (response >= threshold) & (coherence >= 0.52)

        before_bins = np.zeros(work.shape, dtype=np.uint8)
        after_close = np.zeros(work.shape, dtype=np.uint8)
        total_components = 0
        size_filtered_components = 0
        for bin_index in range(18):
            target = -math.pi / 2.0 + (bin_index + 0.5) * math.pi / 18.0
            distance = np.abs(np.angle(np.exp(2j * (orientation - target)))) / 2.0
            mask = (base_mask & (distance <= math.radians(7.0))).astype(np.uint8)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            before_bins = np.maximum(before_bins, mask)
            after_close = np.maximum(after_close, closed)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_components += len(contours)
            for contour in contours:
                if cv2.contourArea(contour) < 45:
                    continue
                rect = cv2.minAreaRect(contour)
                long_side, short_side = max(rect[1]), min(rect[1])
                if short_side < 6 or long_side < 24 or long_side / (short_side + bl.EPS) < 1.25:
                    continue
                size_filtered_components += 1

        orientation_hsv = np.zeros((*work.shape, 3), dtype=np.uint8)
        orientation_hsv[..., 0] = np.mod((orientation + math.pi / 2) * 180 / math.pi, 180).astype(np.uint8)
        orientation_hsv[..., 1] = 255
        orientation_hsv[..., 2] = normalize_for_display(response)
        orientation_rgb = cv2.cvtColor(orientation_hsv, cv2.COLOR_HSV2RGB)

        records.append(
            {
                "local_size": local_size,
                "pool_size": pool_size,
                "threshold": threshold,
                "high_energy": high,
                "coherence": thumbnail(coherence),
                "response": thumbnail(response),
                "orientation_rgb": thumbnail(orientation_rgb),
                "base_mask": thumbnail(base_mask.astype(np.uint8), nearest=True),
                "before_bins": thumbnail(before_bins, nearest=True),
                "after_close": thumbnail(after_close, nearest=True),
                "raw_components": total_components,
                "size_filtered_components": size_filtered_components,
            }
        )
    return work, records


if RUN_OPTIONAL_TENSOR:
    tensor_work, tensor_records = trace_tensor_maps(gray)
    tensor_candidates = bl.tensor_proposals(gray)
else:
    tensor_work, tensor_records, tensor_candidates = gray, [], []

print(f"Tensor working image: {tensor_work.shape[1]} x {tensor_work.shape[0]}")
print(f"Actual tensor_proposals() output: {len(tensor_candidates)} candidate(s)")
display(
    Markdown(
        """
**How to read the tensor summary table:** each row is one pair of neighborhood
sizes. `local_window` measures local orientation, while `pool_window` combines
that evidence over a larger area. The threshold is computed from the page's own
response distribution. Component counts shrink after minimum area, size, and
shape tests are applied.
"""
    )
)
display(
    pd.DataFrame(
        [
            {
                "local_window": item["local_size"],
                "pool_window": item["pool_size"],
                "response_threshold": round(item["threshold"], 4),
                "energy_reference": round(item["high_energy"], 2),
                "orientation_bin_components": item["raw_components"],
                "components_after_size_shape_filters": item["size_filtered_components"],
            }
            for item in tensor_records
        ]
    )
)

# %% [markdown]
# ### Exact tensor stages at each of the three scale pairs
#
# From left to right:
#
# 1. directional coherence;
# 2. pooled barcode response;
# 3. dominant orientation encoded as color;
# 4. energy/coherence threshold;
# 5. separation into 18 orientation bins;
# 6. 5 x 5 closing used only to connect small gaps.

# %%
if tensor_records:
    explain_visual(
        "Figure 10 - exact structure-tensor stages",
        [
            "Read one row from left to right. The three rows repeat the same calculations using increasingly larger local/pooling windows.",
            "Coherence: bright means local edges agree on a direction. Tables can also be bright here, so this column alone is not a barcode answer.",
            "Pooled response: brighter areas combine strong edges and directional agreement over a neighborhood.",
            "Orientation: color represents dominant direction; consistent color inside a region means its edges align.",
            "The last three columns are binary masks: white survives, black is rejected. Orientation bins prevent differently rotated neighboring regions from merging. The 5x5 close only fills tiny gaps.",
        ],
        "The long VIN barcode becomes the strongest white horizontal band in every row. The lower tilted barcode also appears as a smaller component. Much of the table is visible in early maps but does not dominate the final masks.",
        "This figure explains both the tensor route's rotation tolerance and its false-positive risk: tables also contain coherent lines, so tensor regions still need later barcode verification.",
    )

    fig, axes = plt.subplots(len(tensor_records), 6, figsize=(21, 11))
    for row, item in enumerate(tensor_records):
        panels = [
            (item["coherence"], "coherence", "viridis"),
            (item["response"], "pooled response", "magma"),
            (item["orientation_rgb"], "orientation", None),
            (item["base_mask"], "threshold mask", "gray"),
            (item["before_bins"], "18 orientation bins", "gray"),
            (item["after_close"], "after 5x5 close", "gray"),
        ]
        for col, (panel, title, cmap) in enumerate(panels):
            axes[row, col].imshow(panel, cmap=cmap)
            axes[row, col].set_title(
                f"L={item['local_size']}, P={item['pool_size']}\n{title}"
            )
            axes[row, col].axis("off")
    save_and_show(fig, "10-tensor-internal-stages.png")

# %%
display(detections_dataframe(tensor_candidates, "T"))
explain_visual(
    "Figure 11 - raw tensor proposals",
    [
        "Each magenta outline is a quadrilateral returned by the independent tensor route.",
        "Overlapping outlines may be repeated evidence for one barcode. Outlines elsewhere may be table/text structures that only look directionally coherent.",
        "Do not interpret magenta as accepted or decoded; these are hypotheses waiting for deduplication and structural verification.",
    ],
    "The tensor route finds the real stripe groups but returns more raw candidate evidence than the final result needs.",
    "This is why tensor is an optional high-recall proposal source rather than the final decision maker.",
)

show_image(
    draw_detections(image, tensor_candidates, color_by_source=True),
    "Step 2C - Actual optional tensor proposals (magenta)",
    "11-tensor-proposals.png",
    figsize=(11, 15),
)

# %% [markdown]
# # Step 4 - Combine and remove duplicate proposals
#
# The standard path combines ZXing and OpenCV. The tensor route is shown as a
# separate high-recall comparison because it is disabled by default.
#
# Multiple passes often find the same physical barcode. `deduplicate` ranks
# candidates, compares oriented-polygon containment, center distance and angle,
# then merges equivalent candidates while preserving all source names.

# %%
standard_raw = copy.deepcopy(zxing_candidates + opencv_candidates)
high_recall_raw = copy.deepcopy(zxing_candidates + opencv_candidates + tensor_candidates)

print(f"Standard raw candidates:    {len(standard_raw)}")
print(f"With optional tensor route: {len(high_recall_raw)}")

explain_visual(
    "Figure 12 - standard proposal sources together",
    [
        "Green means a region already decoded by whole-page ZXing; amber means a visual proposal from OpenCV.",
        "Several amber outlines can surround one green outline because many OpenCV settings rediscovered the same code.",
        "This figure uses the standard path only. Optional tensor candidates are deliberately excluded from this merge example.",
    ],
    "The two upper barcodes have green semantic anchors plus OpenCV support. The lower barcode initially has OpenCV support but no whole-page ZXing anchor.",
    "Combining sources lets decoding provide strong confirmation without making decoder success mandatory.",
)

show_image(
    draw_detections(image, standard_raw, color_by_source=True, thickness=3),
    "Step 4 - Standard raw candidates: green ZXing, amber OpenCV",
    "12-standard-candidates-before-dedup.png",
    figsize=(11, 15),
)

# %% [markdown]
# ### Which proposals belong to the same physical region?
#
# The table below builds connected groups using exactly the implementation's
# `similar_detection` predicate. Large groups are expected: the same barcode may
# be found by many scales, thresholds and rotations.

# %%
def similarity_clusters(detections: Sequence[bl.Detection]) -> list[list[int]]:
    parents = list(range(len(detections)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parents[root_b] = root_a

    for left in range(len(detections)):
        for right in range(left + 1, len(detections)):
            if bl.similar_detection(detections[left], detections[right]):
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(detections)):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda group: min(group))


clusters = similarity_clusters(standard_raw)
cluster_rows = []
for cluster_id, members in enumerate(clusters, start=1):
    member_detections = [standard_raw[index] for index in members]
    cluster_rows.append(
        {
            "cluster": cluster_id,
            "raw_member_count": len(members),
            "raw_ids": [f"R{index + 1}" for index in members],
            "source_families": sorted({source_family(item) for item in member_detections}),
            "decoded_values": sorted({item.decoded_text for item in member_detections if item.decoded_text}),
            "maximum_pair_containment": round(
                max(
                    [
                        bl.quad_overlap(member_detections[a].quad, member_detections[b].quad)
                        for a in range(len(member_detections))
                        for b in range(a + 1, len(member_detections))
                    ]
                    or [0.0]
                ),
                3,
            ),
        }
    )
display(
    Markdown(
        """
**How to read the similarity-cluster table:** one row represents one physical
region according to the same similarity predicate used by the code. A large
`raw_member_count` means many passes agreed on that region. `decoded_values`
shows whether any member supplied semantic ZXing text. Containment near 1 means
some member polygons almost completely cover one another.
"""
    )
)
display(pd.DataFrame(cluster_rows))

# %%
standard_merged = bl.deduplicate(copy.deepcopy(standard_raw))
high_recall_merged = bl.deduplicate(copy.deepcopy(high_recall_raw))

print(f"Standard:    {len(standard_raw)} raw -> {len(standard_merged)} merged")
print(f"High recall: {len(high_recall_raw)} raw -> {len(high_recall_merged)} merged")
display(detections_dataframe(standard_merged, "M"))

explain_visual(
    "Figure 13 - before and after duplicate removal",
    [
        "Left: all raw standard proposals are drawn, so outlines form thick overlapping stacks.",
        "Right: equivalent proposals have been merged into one object per physical location.",
        "Merging preserves the union of source names and decoded information; it does not yet prove that an undecoded visual region is a barcode.",
    ],
    "The standard path collapses 53 raw proposals into exactly 3 merged regions, matching the three visible barcode locations.",
    "Deduplication converts repeated detector evidence into manageable objects for expensive local verification.",
)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
axes[0].imshow(bgr_to_rgb(draw_detections(image, standard_raw, color_by_source=True, thickness=2)))
axes[0].set_title(f"Before: {len(standard_raw)} overlapping raw proposals")
axes[1].imshow(bgr_to_rgb(draw_detections(image, standard_merged, color_by_source=True, thickness=5)))
axes[1].set_title(f"After: {len(standard_merged)} merged candidates")
for axis in axes:
    axis.axis("off")
save_and_show(fig, "13-before-and-after-deduplication.png")

# %% [markdown]
# # Step 5 - Local perspective rectification
#
# A rotated quadrilateral contains four source points. `rectify_quad` maps them
# to the corners of a normal rectangle using a perspective transform. It also
# rotates portrait crops so every linear candidate ends up wide with vertical
# bars.

# %%
explain_visual(
    "Figure 14 - perspective rectification of every merged candidate",
    [
        "Each row belongs to one merged candidate M1, M2, or M3.",
        "Left is an ordinary axis-aligned neighborhood from the original page; it retains page rotation and surrounding context.",
        "Right maps the candidate's four corners to a normal rectangle. The crop is intentionally horizontal and the barcode bars should be vertical.",
        "Small stretching is expected because perspective warping resamples pixels.",
    ],
    "M3 is tilted in the original page neighborhood but becomes a clean horizontal bar field on the right. M1 and M2 are also normalized to the same analysis convention.",
    "After rectification, one set of left-to-right scanline rules can analyze all barcode angles.",
)

rectified_patches: list[np.ndarray] = []
fig, axes = plt.subplots(len(standard_merged), 2, figsize=(16, 4.5 * len(standard_merged)))
axes = np.atleast_2d(axes)
for index, detection in enumerate(standard_merged, start=1):
    local_view = crop_around_quad(image, detection.quad, padding=50)
    patch = bl.rectify_quad(image, detection.quad, pad_long=0.08, pad_short=0.18)
    rectified_patches.append(patch)

    axes[index - 1, 0].imshow(bgr_to_rgb(local_view))
    axes[index - 1, 0].set_title(f"M{index}: original page neighborhood")
    axes[index - 1, 1].imshow(bgr_to_rgb(patch))
    axes[index - 1, 1].set_title(
        f"M{index}: rectified crop, {patch.shape[1]} x {patch.shape[0]} px"
    )
    for axis in axes[index - 1]:
        axis.axis("off")
save_and_show(fig, "14-all-candidates-rectified.png")

# %% [markdown]
# # Step 6 - Structural verification
#
# Each candidate is rectified **without padding** for measurement. We then test:
#
# 1. horizontal gradient energy dominates after rectification;
# 2. edge columns persist through many rows;
# 3. a median scanline contains enough alternating dark/light runs;
# 4. several independent scanlines agree near transition positions;
# 5. the candidate is materially longer than it is tall.
#
# The next helper mirrors `barcode_structure_metrics` and additionally returns
# every intermediate array for visualization.

# %%
def trace_structural_metrics(input_gray: np.ndarray, quad: np.ndarray) -> dict[str, Any]:
    patch = bl.as_gray(bl.rectify_quad(input_gray, quad, pad_long=0.0, pad_short=0.0))
    height, width = patch.shape[:2]
    top = max(0, int(round(height * 0.10)))
    bottom = min(height, max(top + 3, int(round(height * 0.82))))
    strip = patch[top:bottom]
    if strip.shape[0] < 3:
        strip = patch

    gx = np.abs(cv2.Scharr(strip, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Scharr(strip, cv2.CV_32F, 0, 1))
    x_energy = float(np.mean(gx))
    y_energy = float(np.mean(gy))
    orientation_score = x_energy / (x_energy + y_energy + bl.EPS)

    edge_threshold = float(np.percentile(gx, 72.0))
    if edge_threshold <= bl.EPS:
        edge_threshold = float(np.mean(gx) + np.std(gx))
    edge_mask = gx >= max(edge_threshold, bl.EPS)
    column_support = np.mean(edge_mask, axis=0)
    persistent_columns = column_support >= 0.43
    persistence = float(np.mean(persistent_columns))
    support_strength = (
        float(np.mean(column_support[persistent_columns])) if np.any(persistent_columns) else 0.0
    )

    consensus = np.median(strip, axis=0).astype(np.uint8)
    binary = bl.binary_signal(consensus)
    transitions, dark_fraction = bl.count_transition_runs(binary)
    transition_rate = transitions / max(1, width)
    transition_sites = np.flatnonzero(np.diff(binary.astype(np.int8)) != 0)

    expanded = np.zeros(width, dtype=bool)
    scanline_agreements: list[float] = []
    rows = np.linspace(0, strip.shape[0] - 1, min(9, strip.shape[0]), dtype=int)
    scanline_binaries: list[np.ndarray] = []
    if len(transition_sites):
        for offset in range(-2, 3):
            expanded[np.clip(transition_sites + offset, 0, width - 1)] = True
        for row in rows:
            row_binary = bl.binary_signal(strip[row])
            scanline_binaries.append(row_binary)
            if len(row_binary) == len(binary) and np.any(expanded):
                scanline_agreements.append(float(np.mean(row_binary[expanded] == binary[expanded])))
    agreement = float(np.median(scanline_agreements)) if scanline_agreements else 0.0

    aspect = width / max(1.0, height)
    aspect_score = min(1.0, max(0.0, (aspect - 1.15) / 2.2))
    alternation_score = min(1.0, transitions / 18.0) * min(1.0, transition_rate / 0.055)
    persistence_score = min(1.0, persistence / 0.16) * min(1.0, support_strength / 0.62)
    dark_score = max(0.0, 1.0 - abs(dark_fraction - 0.43) / 0.43)
    contributions = {
        "orientation x 0.22": 0.22 * orientation_score,
        "persistence x 0.30": 0.30 * persistence_score,
        "alternation x 0.25": 0.25 * alternation_score,
        "agreement x 0.16": 0.16 * agreement,
        "aspect x 0.04": 0.04 * aspect_score,
        "dark fraction x 0.03": 0.03 * dark_score,
    }

    reference_metrics = bl.barcode_structure_metrics(input_gray, quad)
    return {
        "patch": patch,
        "strip": strip,
        "gx": gx,
        "gy": gy,
        "edge_threshold": edge_threshold,
        "column_support": column_support,
        "persistent_columns": persistent_columns,
        "consensus": consensus,
        "binary": binary,
        "transition_sites": transition_sites,
        "scanline_rows": rows,
        "scanline_binaries": np.asarray(scanline_binaries),
        "scanline_agreements": scanline_agreements,
        "contributions": contributions,
        "reference_metrics": reference_metrics,
    }


structural_traces = [trace_structural_metrics(gray, item.quad) for item in standard_merged]
for detection, trace_item in zip(standard_merged, structural_traces):
    detection.metrics = trace_item["reference_metrics"]
    detection.structural_score = detection.metrics.get("score", 0.0)

# %% [markdown]
# ### All numerical metrics and their acceptance limits

# %%
metric_rows = []
gate_rows = []
for index, (detection, trace_item) in enumerate(zip(standard_merged, structural_traces), start=1):
    metrics = detection.metrics
    long_side, short_side = detection.long_short()
    aspect_box = long_side / (short_side + bl.EPS)
    has_opencv = any(source.startswith("opencv:") for source in detection.sources)
    metric_rows.append(
        {
            "id": f"M{index}",
            "score": metrics.get("score"),
            "orientation": metrics.get("orientation"),
            "persistence": metrics.get("persistence"),
            "support_strength": metrics.get("support_strength"),
            "scanline_agreement": metrics.get("scanline_agreement"),
            "transitions": metrics.get("transitions"),
            "transition_rate": metrics.get("transition_rate"),
            "dark_fraction": metrics.get("dark_fraction"),
            "rectified_aspect": metrics.get("aspect"),
        }
    )
    gate_rows.append(
        {
            "id": f"M{index}",
            "already_decoded_bypass": bool(detection.barcode_format),
            "score >= 0.52": metrics.get("score", 0.0) >= MINIMUM_SCORE,
            "orientation >= 0.62": metrics.get("orientation", 0.0) >= 0.62,
            "persistence >= 0.045": metrics.get("persistence", 0.0) >= 0.045,
            "transitions >= 12": metrics.get("transitions", 0.0) >= 12,
            "agreement >= 0.50": metrics.get("scanline_agreement", 0.0) >= 0.50,
            "rate in [0.015, 0.72]": 0.015 <= metrics.get("transition_rate", 0.0) <= 0.72,
            "long side >= 90": long_side >= MINIMUM_LINEAR_LENGTH,
            "box aspect >= 1.65": aspect_box >= 1.65,
            "has OpenCV corroboration": has_opencv,
        }
    )

metrics_table = pd.DataFrame(metric_rows).round(4)
gates_table = pd.DataFrame(gate_rows)
display(
    Markdown(
        """
**How to read the metric table:** higher `score`, `orientation`, `persistence`,
`support_strength`, and `scanline_agreement` are better. `transitions` is the
count of meaningful black/white changes. `transition_rate` normalizes that count
by crop width. `dark_fraction` is descriptive rather than a hard gate.

**How to read the gate table:** every column is a yes/no requirement. An already
decoded linear barcode may bypass visual gates because a valid symbolic decode
is stronger evidence. For an undecoded region, all physical gates and OpenCV
corroboration must pass in normal mode.
"""
    )
)
display(metrics_table)
display(gates_table)

# %% [markdown]
# ### Visual audit of every candidate's structural calculation
#
# Each candidate gets its own presentation-ready figure:
#
# - rectified crop and measured middle strip;
# - horizontal edge strength;
# - persistence of edge columns;
# - consensus signal and binary alternating runs;
# - agreement of independent scanlines;
# - exact contribution of every term to the final score.

# %%
for index, (detection, trace_item) in enumerate(zip(standard_merged, structural_traces), start=1):
    metrics = detection.metrics
    explain_visual(
        f"Figure 15.{index} - structural audit for candidate M{index}",
        [
            "Top row: the full rectified candidate is on the left; the middle-height strip actually measured is on the right. The strip avoids most characters printed below the bars.",
            "Second row left: bright vertical lines in `|Gx|` are bar edges. Second row right: blue is the fraction of rows supporting an edge at each x position; the red dashed line is the 0.43 persistence threshold; green fill marks columns that pass.",
            "Third row left: gray is the median grayscale scanline; red is its binary dark/light interpretation; faint vertical markers identify transition positions.",
            "Third row right: each horizontal row is an independent scanline. Vertically aligned black runs mean the rows agree on the same barcode pattern.",
            "Bottom: each colored bar is one weighted contribution. Their sum is the structural score; the maximum possible sum is 1.",
        ],
        f"M{index} has score {metrics['score']:.4f}, {int(metrics['transitions'])} transitions, orientation {metrics['orientation']:.3f}, persistence {metrics['persistence']:.3f}, and scanline agreement {metrics['scanline_agreement']:.3f}. Its initial decoded value is {detection.decoded_text!r}.",
        "The audit makes the acceptance explainable: a true barcode supplies long persistent edge columns, many alternating runs, and agreement across height rather than merely looking rectangular.",
    )

    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.1])

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(trace_item["patch"], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"M{index} - rectified candidate used for measurement")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(trace_item["strip"], cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.set_title("Middle strip: avoids most text below the bars")
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(normalize_for_display(trace_item["gx"]), cmap="magma", aspect="auto")
    ax.set_title(
        f"|Gx|: vertical bar edges; orientation score = {metrics['orientation']:.3f}"
    )
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 1])
    support = trace_item["column_support"]
    ax.plot(support, color="#2563eb", linewidth=1)
    ax.axhline(0.43, color="#dc2626", linestyle="--", label="persistent threshold = 0.43")
    ax.fill_between(
        np.arange(len(support)),
        0,
        support,
        where=trace_item["persistent_columns"],
        color="#16a34a",
        alpha=0.35,
        label="persistent edge columns",
    )
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Column persistence = {metrics['persistence']:.3f}")
    ax.set_xlabel("x position in rectified candidate")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(grid[2, 0])
    ax.plot(trace_item["consensus"], color="#4b5563", linewidth=1, label="median grayscale")
    ax.step(
        np.arange(len(trace_item["binary"])),
        trace_item["binary"].astype(float) * 255,
        where="mid",
        color="#dc2626",
        linewidth=0.8,
        alpha=0.75,
        label="binary dark/light signal",
    )
    for site in trace_item["transition_sites"]:
        ax.axvline(site, color="#f59e0b", alpha=0.12, linewidth=0.7)
    ax.set_ylim(-10, 265)
    ax.set_title(
        f"Consensus signal: {int(metrics['transitions'])} alternating transitions"
    )
    ax.legend(loc="upper right")

    ax = fig.add_subplot(grid[2, 1])
    scanlines = trace_item["scanline_binaries"]
    if scanlines.size:
        ax.imshow(scanlines, cmap="gray_r", aspect="auto", interpolation="nearest")
    ax.set_title(f"Independent scanlines; median agreement = {metrics['scanline_agreement']:.3f}")
    ax.set_xlabel("x position")
    ax.set_ylabel("sampled row")

    ax = fig.add_subplot(grid[3, :])
    contributions = trace_item["contributions"]
    names = list(contributions)
    values = list(contributions.values())
    bars = ax.barh(names, values, color=["#2563eb", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#6b7280"])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_xlim(0, max(0.34, max(values) * 1.2))
    ax.set_xlabel("Contribution to final score")
    ax.set_title(
        f"Weighted score = {sum(values):.4f}; implementation score = {metrics['score']:.4f}"
    )

    fig.suptitle(
        f"Candidate M{index}: decoded initially = {detection.decoded_text!r}",
        fontsize=16,
        y=1.01,
    )
    save_and_show(fig, f"15-structural-audit-M{index}.png")

# %% [markdown]
# # Step 7 - Local ZXing decoding of rectified candidates
#
# `decode_locally` adds 8% padding along the long axis, 24% along the short axis,
# adds a white border, and calls ZXing. Decoding can increase confidence but is
# not required for a structurally valid visual detection.
#
# On page 6, the first two codes are already decoded by the whole-page ZXing
# pass. The lower component label demonstrates the value of local rectification:
# it is found visually and then decoded from its cleaned crop.

# %%
pipeline_candidates = copy.deepcopy(standard_merged)
decode_rows = []
decode_patches = []
accepted: list[bl.Detection] = []
rejected: list[bl.Detection] = []

for index, candidate in enumerate(pipeline_candidates, start=1):
    before_text = candidate.decoded_text
    before_format = candidate.barcode_format

    if candidate.kind == "matrix" and candidate.barcode_format:
        candidate.accepted = True
        accepted.append(candidate)
        decode_rows.append(
            {
                "id": f"M{index}",
                "attempted_local_decode": False,
                "reason": "decoded matrix anchor",
                "before": before_text,
                "after": candidate.decoded_text,
                "format": candidate.barcode_format,
                "accepted": True,
            }
        )
        continue

    candidate.metrics = bl.barcode_structure_metrics(gray, candidate.quad)
    candidate.structural_score = candidate.metrics.get("score", 0.0)
    should_decode = (
        candidate.structural_score >= max(0.35, MINIMUM_SCORE - 0.16)
        or bool(candidate.barcode_format)
    )

    local_patch = bl.rectify_quad(image, candidate.quad, pad_long=0.08, pad_short=0.24)
    local_patch = cv2.copyMakeBorder(local_patch, 16, 16, 20, 20, cv2.BORDER_CONSTANT, value=255)
    decode_patches.append((f"M{index}", local_patch, should_decode))

    if should_decode:
        bl.decode_locally(image, candidate)

    candidate.accepted = bl.accept_linear(
        candidate,
        MINIMUM_SCORE,
        allow_tensor_only=False,
        minimum_linear_length=MINIMUM_LINEAR_LENGTH,
    )
    (accepted if candidate.accepted else rejected).append(candidate)
    decode_rows.append(
        {
            "id": f"M{index}",
            "attempted_local_decode": should_decode,
            "reason": "score high enough" if should_decode else "score below local-decode gate",
            "before": before_text,
            "before_format": before_format,
            "after": candidate.decoded_text,
            "format": candidate.barcode_format,
            "structural_score": round(candidate.structural_score, 4),
            "accepted": candidate.accepted,
        }
    )

display(
    Markdown(
        """
**How to read the local-decoding table:** `before` is text already known from the
whole-page ZXing stage. `attempted_local_decode` says whether the structural gate
allowed a crop retry. `after` is the text stored after that retry. The important
page-6 case is the lower candidate: its `before` value is empty, but its
rectified local crop produces a valid `after` value.
"""
    )
)
display(pd.DataFrame(decode_rows))

explain_visual(
    "Figure 16 - exact crops sent to local ZXing",
    [
        "Each horizontal panel is one rectified candidate with extra white border. The border supplies a quiet zone so the decoder can see where the symbol begins and ends.",
        "Read the panel title: it reports the candidate ID, whether a decode was attempted, and the returned text.",
        "The three crops are not required to have the same height or width because the printed symbols have different physical dimensions.",
    ],
    "M1 and M2 already had whole-page decoded values. M3 had no whole-page result, but its straightened crop decodes as `20X7010078066`.",
    "Local rectification reduces the decoder's search problem from a complete noisy A3 page to one clean, correctly oriented symbol.",
)

fig, axes = plt.subplots(len(decode_patches), 1, figsize=(16, 3.2 * len(decode_patches)))
axes = np.atleast_1d(axes)
for axis, (candidate_id, patch, attempted) in zip(axes, decode_patches):
    axis.imshow(bgr_to_rgb(patch))
    result = next(row for row in decode_rows if row["id"] == candidate_id)
    axis.set_title(
        f"{candidate_id}: patch sent to ZXing = {attempted}; result = {result.get('after')!r}"
    )
    axis.axis("off")
save_and_show(fig, "16-local-zxing-inputs-and-results.png")

# %% [markdown]
# # Step 8 - Final accepted detections and output contract
#
# The final output preserves oriented quadrilaterals. Green means decoded;
# amber means visually accepted but not decoded; magenta means a decoded matrix
# symbol.

# %%
accepted = bl.deduplicate(accepted)
accepted.sort(key=lambda item: (round(float(item.center()[1]) / 20), float(item.center()[0])))

final_overlay = bl.annotate(image, accepted)
explain_visual(
    "Figure 17 - final production overlay",
    [
        "A green quadrilateral means the region was accepted and decoded. Amber would mean accepted from visual structure but not decoded. Magenta would mean a decoded matrix symbol.",
        "The number is the final top-to-bottom result order. The text beside it is the decoded value.",
        "Notice that boxes are rotated quadrilaterals following the barcode itself, not large horizontal rectangles containing excess page content.",
    ],
    "Exactly three final green boxes surround the small header code, long VIN code, and tilted lower component code. No table cell or ordinary text is accepted.",
    "This is the operational output a human reviewer sees; coordinates and metrics are preserved separately in JSON.",
)

show_image(
    final_overlay,
    f"Final page-6 output: {len(accepted)} accepted, {sum(bool(x.barcode_format) for x in accepted)} decoded",
    "17-final-page-06-overlay.png",
    figsize=(11, 15),
)

final_table = detections_dataframe(accepted, "A")
if not final_table.empty:
    final_table["accepted"] = [item.accepted for item in accepted]
    final_table["json_confidence"] = [
        round(max(item.proposal_score, item.structural_score), 4) for item in accepted
    ]
display(
    Markdown(
        """
**How to read the final table:** one row equals one final barcode. The center,
angle, and side lengths describe its geometry in page pixels. `proposal_score`
describes source evidence; `structural_score` describes visible barcode
structure; `json_confidence` is their maximum. It is a deterministic evidence
score, not a statistically calibrated probability.
"""
    )
)
display(final_table)

if rejected:
    display(Markdown("### Rejected candidates"))
    display(detections_dataframe(rejected, "R"))
else:
    print("No merged standard candidate was rejected on page 6.")

# %% [markdown]
# ### Deskewed final crops

# %%
explain_visual(
    "Figure 18 - final deskewed crops",
    [
        "Each row corresponds to one final result A1, A2, or A3 from the final table.",
        "All page context has intentionally been removed. The bars are normalized to vertical orientation inside a wide horizontal crop.",
        "Use these crops for visual quality review or a later decoder retry; use the oriented `quad` in JSON when mapping back to the page.",
    ],
    "All three crops contain clean repeated bars. The lower tilted page label now looks as easy to inspect as the two horizontal labels.",
    "The crop gallery separates `where it was on the page` from `what the barcode pixels look like after geometric normalization`.",
)

fig, axes = plt.subplots(len(accepted), 1, figsize=(16, 3.2 * len(accepted)))
axes = np.atleast_1d(axes)
for index, (axis, detection) in enumerate(zip(axes, accepted), start=1):
    patch = bl.rectify_quad(image, detection.quad, pad_long=0.08, pad_short=0.18)
    axis.imshow(bgr_to_rgb(patch))
    axis.set_title(
        f"A{index}: {detection.barcode_format or 'visual-only'} - {detection.decoded_text!r}"
    )
    axis.axis("off")
save_and_show(fig, "18-final-deskewed-crops.png")

# %% [markdown]
# ### JSON-like machine-readable result
#
# `quad` is the accurate rotated box. `aabb` is included only for systems that
# cannot consume oriented polygons. For undecoded detections, `confidence` is a
# deterministic evidence score, not a calibrated probability.

# %%
page_payload = {
    "page": PAGE_NUMBER,
    "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
    "detections": [item.to_json() for item in accepted],
}
display(
    Markdown(
        """
**How to read the JSON:** `quad` contains four accurate rotated corner points in
page pixels. `aabb` is the larger axis-aligned compatibility box. `decoded`
contains text and symbology when available. `sources` records which independent
passes supported the object, and `structural_metrics` preserves the numerical
explanation used during verification.
"""
    )
)
print(json.dumps(page_payload, indent=2, ensure_ascii=False))

with (OUTPUT_DIR / "page-06-walkthrough-result.json").open("w", encoding="utf-8") as handle:
    json.dump(page_payload, handle, indent=2, ensure_ascii=False)

# %% [markdown]
# ## Exact reference check against `locate_page`
#
# This final call runs the unmodified public orchestration function. It verifies
# that the teaching reconstruction produces the same accepted count and decoded
# values as the source implementation.

# %%
reference_accepted, reference_rejected = bl.locate_page(
    image,
    angle_step=ANGLE_STEP,
    minimum_score=MINIMUM_SCORE,
    use_tensor_fallback=False,
    allow_tensor_only=False,
    minimum_linear_length=MINIMUM_LINEAR_LENGTH,
)

comparison = {
    "walkthrough_accepted_count": len(accepted),
    "reference_accepted_count": len(reference_accepted),
    "walkthrough_decoded": sorted(item.decoded_text for item in accepted if item.decoded_text),
    "reference_decoded": sorted(item.decoded_text for item in reference_accepted if item.decoded_text),
    "reference_rejected_count": len(reference_rejected),
}
print(json.dumps(comparison, indent=2))

assert comparison["walkthrough_accepted_count"] == comparison["reference_accepted_count"]
assert comparison["walkthrough_decoded"] == comparison["reference_decoded"]
print("PASS: the walkthrough and barcode_locator.locate_page agree on page 6.")

# %% [markdown]
# # Slide-ready summary
#
# ```text
# private evaluation document PDF
#         |
#         v
# page 6 raster at 200 DPI
#         |
#         +--> ZXing whole-page decoded anchors
#         |
#         +--> OpenCV directional-coherence detector
#         |       - 3 native scale profiles
#         |       - thresholds 48 and 64
#         |       - rotations 15, 30, 45, 60, 75 degrees
#         |
#         +--> optional independent structure tensor
#                     |
#                     v
#         merge duplicate oriented polygons
#                     |
#                     v
#         rectify each quadrilateral
#                     |
#                     v
#         persistent edges + alternating runs
#         + multi-scanline agreement + shape gates
#                     |
#                     v
#         optional local ZXing decoding
#                     |
#                     v
#         oriented boxes + crops + JSON
# ```
#
# ## Main management message
#
# The system does not trust one fragile image filter. It collects several
# deterministic pieces of evidence, normalizes each candidate's geometry, then
# verifies physical barcode properties before producing a result. No detector
# has been trained on the evaluation documents.
#
# ## Research basis
#
# - OpenCV's classical 1-D barcode algorithm: directional coherence of gradient
#   patches and connection of similar orientations.
# - G. Sörös and C. Flörkemeier, *Blur-Resistant Joint 1D and 2D Barcode
#   Localization for Smartphones*, MUM 2013.
# - G. Sörös, *GPU-Accelerated Joint 1D and 2D Barcode Localization on
#   Smartphones*, ICASSP 2014.
# - Standard projective geometry for quadrilateral rectification.
# - ZXing-C++ for deterministic open-source decoding.

# %%
print("Presentation figures written to:")
for path in sorted(FIGURE_DIR.glob("*.png")):
    print(" -", path.name)
print("\nNotebook walkthrough complete.")
