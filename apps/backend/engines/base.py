"""Common vocabulary every production barcode engine speaks.

The extraction and localization implementations have different native signatures.
Nothing in the web layer should know that: registry adapters convert their results to
these types, and everything above the registry consumes only these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


class Capability(str, Enum):
    """What an engine is able to tell you about a page."""

    DETECT = "detect"
    """Locates barcodes. Returns geometry only — never a payload."""

    DECODE = "decode"
    """Locates *and* reads barcodes. Returns geometry plus a payload."""


@dataclass(frozen=True)
class Roi:
    """A region of interest as fractions of page width/height.

    Stored normalised so one box drawn on one page applies to every page in a batch
    regardless of size or orientation. Strict by contract: whatever falls outside is
    never examined, which is the entire source of the speed-up.
    """

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("w", self.w), ("h", self.h)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"roi.{name} must be within [0,1], got {value}")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("roi width and height must be positive")
        if self.x + self.w > 1.0 + 1e-9 or self.y + self.h > 1.0 + 1e-9:
            raise ValueError("roi extends past the page edge")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return (left, top, right, bottom) clamped to the image, guaranteed non-empty."""
        left = int(round(self.x * width))
        top = int(round(self.y * height))
        right = int(round((self.x + self.w) * width))
        bottom = int(round((self.y + self.h) * height))
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        return left, top, right, bottom

    def crop(self, image: np.ndarray, margin: int = 0) -> tuple[np.ndarray, int, int]:
        """Crop and return the view plus the (dx, dy) needed to map coordinates back.

        `margin` widens the crop by that many pixels on every side without changing what
        the zone means. A decoder needs the quiet zone around a symbol, so a barcode
        sitting near the zone edge fails to decode when the crop cuts flush against it —
        the extra context is read, but only symbols inside the true zone are kept.
        """
        height, width = image.shape[:2]
        left, top, right, bottom = self.to_pixels(width, height)
        if margin > 0:
            left = max(0, left - margin)
            top = max(0, top - margin)
            right = min(width, right + margin)
            bottom = min(height, bottom + margin)
        return image[top:bottom, left:right], left, top

    def contains_point(self, x: float, y: float, width: int, height: int) -> bool:
        """Whether a full-page pixel coordinate falls inside the strict zone."""
        left, top, right, bottom = self.to_pixels(width, height)
        return left <= x <= right and top <= y <= bottom


# Pixel padding added around a zone crop purely so edge symbols keep their quiet zone.
# 64px is a little over the widest quiet zone the supported symbologies ask for at the
# 200-300 ppi this app rasterises to.
CONTEXT_MARGIN_PX = 64


def region_center(quad: tuple[float, ...]) -> tuple[float, float]:
    xs = quad[0::2]
    ys = quad[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


class RegionStatus(str, Enum):
    """How much the engine was able to establish about a located symbol.

    Failing to decode is a result, not an absence of one: an operator needs to see that a
    barcode is there but unreadable, which is a different problem from no barcode at all.
    """

    DECODED = "decoded"
    """Located and read."""

    UNRESOLVED_MATRIX = "localized_unresolved_matrix"
    """A 2-D symbol was located and confidently identified, but would not decode."""

    UNRESOLVED_LINEAR = "localized_unresolved_linear"
    """A linear symbol was physically confirmed, but its payload would not decode."""

    REVIEW_CANDIDATE = "review_candidate"
    """Something barcode-like was located; weaker evidence, worth a human look."""

    LOCALIZED = "localized"
    """Located by a detection-only engine, which never attempts a payload."""


@dataclass(frozen=True)
class Region:
    """One located barcode.

    `value` and `symbology` are None for every DETECT-capability engine. That is not a
    missing field to be filled in later — a localizer physically cannot produce a payload,
    and the API contract exposes the two cases as different response shapes.

    `extras` carries engine-specific diagnostics that don't fit the common fields — for
    example a vendored decode reports `attempts`, `aabb`, `evidence` and `crop` that the
    UI surfaces verbatim when the operator inspects an unresolved or review candidate.
    """

    quad: tuple[float, float, float, float, float, float, float, float]
    kind: str
    confidence: float
    value: str | None = None
    symbology: str | None = None
    sources: tuple[str, ...] = ()
    status: RegionStatus = RegionStatus.LOCALIZED
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def decoded(self) -> bool:
        return self.value is not None


@dataclass
class PageOutcome:
    page: int
    regions: list[Region] = field(default_factory=list)
    elapsed_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_barcode(self) -> bool:
        return bool(self.regions)


@dataclass(frozen=True)
class EngineInfo:
    """Everything the UI needs to describe an engine without hard-coding it."""

    id: str
    label: str
    capability: Capability
    summary: str
    speed_ms_per_page: str
    accuracy_note: str
    badge: str
    available: bool = True
    unavailable_reason: str | None = None
    license_warning: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Engine(Protocol):
    """The contract every engine implements."""

    info: EngineInfo

    def warm(self) -> None:
        """Load models and prime caches. Called once at startup, never per request.

        Several engines have non-thread-safe first calls, so this must complete before
        any worker thread touches the engine.
        """

    def analyze_page(
        self,
        page: int,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> PageOutcome:
        """Analyze one page image. Must be safe to call from multiple threads."""


@runtime_checkable
class PreparedEngine(Protocol):
    """An engine whose input preparation can happen before its compute timer.

    A queue or upstream service may call :meth:`prepare_page` and pass the returned
    opaque value to :meth:`analyze_prepared`. This keeps file decoding and input
    transforms outside the engine's compute-latency contract.
    """

    info: EngineInfo

    def prepare_page(
        self,
        path: Path,
        *,
        roi: Roi | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Create the engine-ready input. This work is not engine compute."""

    def analyze_prepared(self, page: int, prepared: Any) -> PageOutcome:
        """Analyze an already-prepared input and time only engine compute."""
