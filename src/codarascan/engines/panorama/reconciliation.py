# SPDX-License-Identifier: Apache-2.0
"""Reconcile Panorama observations of physical symbols at the package boundary.

Decoder success is not a calibrated confidence score: a recovery crop can read
its neighbour, and checksum-free short ITF reads can be fragments of other
symbols. Keep spatially separate copies of a payload, prefer measured decoder
positions, and retain unsupported candidates for review. No payload is invented.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zxingcpp

from codarascan.core.contracts import Region, RegionStatus
from codarascan.formats import resolve_format


def _polygon(region: Region) -> np.ndarray:
    return cv2.convexHull(np.asarray(region.quad, np.float32).reshape(4, 2)).reshape(
        -1, 2
    )


def _area(region: Region) -> float:
    return abs(float(cv2.contourArea(_polygon(region))))


def _intersection(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, np.ndarray | None]:
    area, polygon = cv2.intersectConvexConvex(first, second)
    return max(0.0, float(area)), polygon


def _overlap(first: Region, second: Region) -> float:
    area = min(_area(first), _area(second))
    return (
        _intersection(_polygon(first), _polygon(second))[0] / area
        if area > 1e-6
        else 0.0
    )


def _axes(region: Region) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    # Use the convex hull so reversed package winding has the same geometry.
    points = _polygon(region)
    edges = np.roll(points, -1, axis=0) - points
    edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    unit = edge / max(float(np.linalg.norm(edge)), 1e-6)
    if unit[0] < 0 or (abs(unit[0]) < 1e-6 and unit[1] < 0):
        unit = -unit
    normal = np.asarray((-unit[1], unit[0]), np.float32)
    long = points @ unit
    short = points @ normal
    center = (
        unit * (long.min() + long.max()) / 2 + normal * (short.min() + short.max()) / 2
    )
    return center, unit, normal, float(np.ptp(long)), float(np.ptp(short))


def _observed(region: Region) -> bool:
    return any(
        source
        in {
            "panorama:whole-page-zxing",
            "zxing:linear-whole-page",
            "reconciliation:decoder-position",
        }
        for source in region.sources
    )


def _joined(region: Region) -> bool:
    return any("fragment-join" in source for source in region.sources)


def _wide_recovery(region: Region) -> bool:
    return any("wide-parent" in source for source in region.sources)


def _rank(region: Region) -> tuple:
    return (
        not _observed(region),
        _joined(region),
        _wide_recovery(region),
        -region.confidence,
        _area(region),
        tuple(region.quad),
        region.sources,
    )


def _identity(region: Region) -> tuple:
    return region.kind, (region.symbology or "").casefold(), region.value


def _aligned(first: Region, second: Region) -> bool:
    if first.kind != "linear" or second.kind != "linear":
        return True
    return abs(float(_axes(first)[1] @ _axes(second)[1])) >= np.cos(np.deg2rad(15))


def _borrowed(candidate: Region, observed: Region) -> bool:
    # Wide-parent recovery intentionally searches beyond the proposal. A precise
    # matching read outside its centre, but intersecting its edge, owns that read.
    if not _wide_recovery(candidate) or not _observed(observed):
        return False
    center, unit, normal, length, height = _axes(candidate)
    other_center = np.asarray(observed.quad).reshape(4, 2).mean(axis=0)
    offset = other_center - center
    return (
        _overlap(candidate, observed) >= 0.10
        and abs(float(offset @ unit)) > length / 2
        and abs(float(offset @ unit)) <= length
        and abs(float(offset @ normal)) <= height
    )


def _bar_direction(region: Region, gray: np.ndarray) -> np.ndarray | None:
    """Measure the direction across the bars, independently of the box edges."""
    polygon = _polygon(region)
    x, y, width, height = cv2.boundingRect(polygon)
    left, top = max(0, x - 1), max(0, y - 1)
    right, bottom = (
        min(gray.shape[1], x + width + 1),
        min(gray.shape[0], y + height + 1),
    )
    if right - left < 3 or bottom - top < 3:
        return None
    patch = gray[top:bottom, left:right]
    mask = np.zeros(patch.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon - (left, top)).astype(np.int32), 1)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    active = (mask != 0) & (gx * gx + gy * gy >= 80**2)
    if np.count_nonzero(active) < 32:
        return None
    dx, dy = gx[active].astype(np.float64), gy[active].astype(np.float64)
    xx, yy, xy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    coherence = np.hypot(xx - yy, 2 * xy) / max(xx + yy, 1.0)
    if coherence < 0.7:
        return None
    angle = 0.5 * np.arctan2(2 * xy, xx - yy)
    return np.array((np.cos(angle), np.sin(angle)))


def _contained_scanline(
    first: Region,
    second: Region,
    direction: Callable[[Region], np.ndarray | None],
) -> bool:
    """A decoder strip is not an independently oriented physical barcode.

    Require containment, comparable longitudinal extent, and matching *pixel*
    directions. A real crossing label with its own bars must remain separate.
    """
    if first.kind != "linear" or second.kind != "linear":
        return False
    strip, field = sorted((first, second), key=_area)
    if not _observed(strip) or _overlap(strip, field) < 0.8:
        return False
    _, _, _, _, strip_height = _axes(strip)
    center, unit, normal, length, height = _axes(field)
    projected = _polygon(strip) - center
    if not (
        # Decoder support can be a substantial central band, not just a
        # one-pixel line. Judge its extent relative to the containing field.
        strip_height <= 0.6 * height
        and _area(strip) <= 0.6 * _area(field)
        and np.ptp(projected @ unit) >= 0.6 * length
        and abs(float(projected.mean(axis=0) @ normal)) <= 0.3 * height
    ):
        return False
    strip_direction, field_direction = direction(strip), direction(field)
    return (
        strip_direction is not None
        and field_direction is not None
        and abs(float(strip_direction @ field_direction)) >= np.cos(np.deg2rad(8))
    )


def _same_symbol(
    first: Region,
    second: Region,
    direction: Callable[[Region], np.ndarray | None],
) -> bool:
    return _identity(first) == _identity(second) and (
        (_aligned(first, second) and _overlap(first, second) >= 0.45)
        or _contained_scanline(first, second, direction)
        or _borrowed(second, first)
        or _borrowed(first, second)
    )


def _coincident_bar_field(
    first: Region,
    second: Region,
    direction: Callable[[Region], np.ndarray | None],
) -> bool:
    """Test physical coincidence independently of the decoded payload.

    Partial overlap is insufficient: two stickers may really overlap. Require
    comparable full-length footprints centred on the same bar field, supported
    by the image's bar direction. Joined search areas are not physical evidence.
    """
    if first.kind != "linear" or second.kind != "linear":
        return False
    if any(
        (_joined(r) or _wide_recovery(r)) and not _observed(r) for r in (first, second)
    ):
        return False
    center, unit, normal, length, height = _axes(first)
    other_center, other_unit, _, other_length, other_height = _axes(second)
    offset = other_center - center
    intersection = _intersection(_polygon(first), _polygon(second))[0]
    if not (
        abs(float(unit @ other_unit)) >= np.cos(np.deg2rad(8))
        and min(length, other_length) >= 0.8 * max(length, other_length)
        and abs(float(offset @ unit)) <= 0.1 * min(length, other_length)
        and abs(float(offset @ normal)) <= 0.25 * min(height, other_height)
        and intersection >= 0.8 * min(_area(first), _area(second))
        and intersection >= 0.6 * max(_area(first), _area(second))
    ):
        return False
    first_direction, second_direction = direction(first), direction(second)
    return (
        first_direction is not None
        and second_direction is not None
        and abs(float(first_direction @ second_direction)) >= np.cos(np.deg2rad(8))
        and abs(float(first_direction @ unit)) >= np.cos(np.deg2rad(15))
        and abs(float(second_direction @ other_unit)) >= np.cos(np.deg2rad(15))
    )


def _resolve_payload_conflicts(
    regions: list[Region],
    direction: Callable[[Region], np.ndarray | None],
) -> tuple[list[Region], int]:
    """One physical symbol may have several incompatible decode hypotheses.

    Preserve every alternative, but publish no accepted value until they agree.
    Complete-link grouping prevents an intermediate box joining separate labels.
    The earlier identity-based pass already consolidates agreeing observations.
    """
    groups: list[list[Region]] = []
    for region in sorted(regions, key=lambda r: (*_rank(r), _identity(r))):
        group = next(
            (
                g
                for g in groups
                if all(_coincident_bar_field(region, p, direction) for p in g)
            ),
            None,
        )
        if group is None:
            groups.append([region])
        else:
            group.append(region)
    output = []
    conflicts = 0
    for group in groups:
        if len({_identity(r) for r in group}) == 1:
            output.extend(group)
            continue
        conflicts += 1
        representative = group[0]
        formats = {r.symbology for r in group}
        output.append(
            replace(
                representative,
                value=None,
                symbology=representative.symbology if len(formats) == 1 else None,
                status=RegionStatus.UNRESOLVED_LINEAR,
                confidence=0.0,
                sources=tuple(sorted({source for r in group for source in r.sources})),
                extras={
                    **_details(
                        representative,
                        reason="conflicting-decodes",
                        alternatives=[
                            {
                                "value": r.value,
                                "symbology": r.symbology,
                                "confidence": r.confidence,
                                "quad": list(r.quad),
                                "sources": list(r.sources),
                            }
                            for r in sorted(group, key=lambda r: _identity(r))
                        ],
                    ),
                    "decoded": False,
                },
            )
        )
    return output, conflicts


def _physical_geometry(
    observed: Region,
    proposals: list[Region],
    direction: Callable[[Region], np.ndarray | None],
) -> Region:
    """Use a tight, confirmed field around a decoder anchor when available.

    ZXing can report a horizontal strip through a rotated barcode. Its position
    identifies the symbol, while an aligned localizer describes its full extent.
    Joined parents and wide recovery crops are never geometry authorities.
    """
    if not _observed(observed) or observed.kind != "linear":
        return observed
    center, unit, normal, length, height = _axes(observed)
    eligible = []
    for proposal in proposals:
        if _observed(proposal) or _joined(proposal) or _wide_recovery(proposal):
            continue
        other_center, _, _, other_length, other_height = _axes(proposal)
        offset = other_center - center
        if _contained_scanline(observed, proposal, direction) or (
            _aligned(observed, proposal)
            and 0.8 * length <= other_length <= 1.35 * length
            and abs(float(offset @ unit)) <= 0.2 * length
            and abs(float(offset @ normal)) <= 0.5 * max(height, other_height)
        ):
            eligible.append(proposal)
    if not eligible:
        return observed
    best = min(eligible, key=_rank)
    return replace(
        observed,
        quad=best.quad,
        extras=_details(
            observed,
            decoder_quad=list(observed.quad),
            localization_sources=list(best.sources),
            geometry="confirmed-localizer-field",
        ),
    )


def _covered(region: Region, decoded: list[Region]) -> bool:
    area = _area(region)
    if area <= 1e-6:
        return False
    polygon = _polygon(region)
    pieces = []
    for item in decoded:
        if item.kind != region.kind or not _aligned(region, item):
            continue
        amount, clipped = _intersection(polygon, _polygon(item))
        if amount > 1e-6 and clipped is not None:
            pieces.append((amount, clipped.reshape(-1, 2)))
    # Bonferroni lower bound on union coverage. Unlike summing intersections,
    # overlapping decoded symbols cannot manufacture coverage of an unseen area.
    covered = sum(amount for amount, _ in pieces)
    for index, (_, first) in enumerate(pieces):
        for _, second in pieces[index + 1 :]:
            covered -= _intersection(first, second)[0]
    return max([covered, *(amount for amount, _ in pieces)], default=0.0) / area >= 0.60


def _details(region: Region, **details: Any) -> dict[str, Any]:
    return {
        **region.extras,
        "reconciliation": {
            **region.extras.get("reconciliation", {}),
            **details,
        },
    }


def _refine_recovery(region: Region, gray: np.ndarray) -> list[Region]:
    """Map a joined/wide crop's existing payload back to observed page pixels."""
    height, width = gray.shape[:2]
    xs, ys = region.quad[0::2], region.quad[1::2]
    pad_x = max(16, (max(xs) - min(xs)) * (0.5 if _wide_recovery(region) else 0.04))
    pad_y = max(16, (max(ys) - min(ys)) * 0.4)
    left, top = max(0, int(min(xs) - pad_x)), max(0, int(min(ys) - pad_y))
    right, bottom = (
        min(width, int(max(xs) + pad_x + 1)),
        min(height, int(max(ys) + pad_y + 1)),
    )
    if right <= left or bottom <= top:
        return [region]
    try:
        formats = zxingcpp.barcode_formats_from_str(
            resolve_format(region.symbology).zxing_name
        )
        reads = zxingcpp.read_barcodes(
            gray[top:bottom, left:right],
            formats=formats,
            try_rotate=True,
            try_downscale=False,
            try_invert=True,
        )
    except (ValueError, cv2.error):
        return [region]
    matches = []
    for read in reads:
        if not read.valid or read.text != region.value:
            continue
        position = read.position
        points = (
            position.top_left,
            position.top_right,
            position.bottom_right,
            position.bottom_left,
        )
        quad = tuple(
            coordinate
            for point in points
            for coordinate in (float(point.x + left), float(point.y + top))
        )
        measured = replace(
            region,
            quad=quad,
            sources=("reconciliation:decoder-position", *region.sources),
            extras=_details(
                region, original_quad=list(region.quad), geometry="decoder-position"
            ),
        )
        if _area(measured) > 1e-6 and _overlap(region, measured) >= 0.10:
            matches.append(measured)
    return matches or [region]


def _extend_scanline(
    region: Region,
    gray: np.ndarray,
    *,
    partial_field: bool = False,
) -> Region:
    """Recover a decoder band's bar-field height from image evidence.

    An overlaid label may leave only two readable rows. Follow the *same signed
    edges* perpendicular to that row until support ends; never bridge a white
    gap or infer the whole box solely from a neighbouring proposal.
    """
    center, unit, normal, length, height = _axes(region)
    if length < 40 or (not partial_field and height > max(3.0, length * 0.015)):
        return region
    direction = _bar_direction(region, gray)
    if direction is not None and abs(float(direction @ unit)) < np.cos(np.deg2rad(8)):
        # Horizontal decoding bands can cut across rotated bars. Following the
        # rectangle's normal would manufacture an unrelated axis-aligned field;
        # retain the sample for matching to a confirmed rotated localizer instead.
        return region
    # Bound work in samples, not page pixels, so a higher PDF render resolution
    # does not truncate the same physical bar field at an arbitrary pixel height.
    extent = min(float(np.hypot(*gray.shape)), max(length * 0.25, height * 0.75))
    step = max(1.0, length / 1600)
    radius = int(np.ceil(extent / step))
    samples = min(1600, max(40, round(length)))
    along = np.linspace(-length / 2, length / 2, samples, dtype=np.float32)
    across = np.arange(-radius, radius + 1, dtype=np.float32) * step
    points = center + along[None, :, None] * unit + across[:, None, None] * normal
    patch = cv2.remap(
        gray,
        points[:, :, 0],
        points[:, :, 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    ).astype(np.float32)
    gradient = np.diff(patch, axis=1)
    reference = gradient[radius]
    positive, negative = reference > 40, reference < -40
    count = int(positive.sum() + negative.sum())
    if count < 20:
        return region
    kernel = np.ones((1, 7), np.uint8)
    plus = cv2.dilate(np.maximum(gradient, 0), kernel)
    minus = cv2.dilate(np.maximum(-gradient, 0), kernel)
    support = (
        (plus[:, positive] > 40).sum(axis=1) + (minus[:, negative] > 40).sum(axis=1)
    ) / count
    bounds = []
    for direction in (-1, 1):
        last, gap = radius, 0
        for index in range(
            radius + direction, len(across) if direction > 0 else -1, direction
        ):
            if support[index] >= 0.55:
                last, gap = index, 0
            else:
                gap += 1
                if gap >= 3:
                    break
        bounds.append(float(across[last]))
    low, high = bounds
    if high - low <= max(height * (1.1 if partial_field else 2), 6):
        return region
    points = np.array(
        [
            center - length / 2 * unit + low * normal,
            center + length / 2 * unit + low * normal,
            center + length / 2 * unit + high * normal,
            center - length / 2 * unit + high * normal,
        ]
    )
    points[:, 0] = np.clip(points[:, 0], 0, gray.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, gray.shape[0] - 1)
    extended = replace(
        region,
        quad=tuple(float(v) for v in points.reshape(-1)),
        extras=_details(
            region, scanline_quad=list(region.quad), geometry="supported-bar-field"
        ),
    )
    return extended if _area(extended) > 1e-6 else region


def reconcile_regions(
    regions: Iterable[Region],
    *,
    gray: np.ndarray | None = None,
    image_path: Path | None = None,
) -> tuple[list[Region], dict[str, Any]]:
    candidates = list(regions)
    suppressed: list[dict[str, Any]] = []
    revised: list[Region] = []
    geometry_refinements = 0
    short_reads = 0
    directions: dict[tuple[float, ...], np.ndarray | None] = {}

    def pixels() -> np.ndarray | None:
        nonlocal gray
        if gray is None and image_path is not None:
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return gray

    def direction(region: Region) -> np.ndarray | None:
        key = tuple(region.quad)
        if key not in directions:
            image = pixels()
            directions[key] = (
                _bar_direction(region, image) if image is not None else None
            )
        return directions[key]

    for region in candidates:
        if (
            region.status is RegionStatus.DECODED
            and (region.symbology or "").casefold() == "itf"
            and len(region.value or "") < 6
        ):
            # Match the package's Tensor minimum; preserve the rejected hypothesis
            # as review evidence unless a real decoded symbol covers this region.
            short_reads += 1
            revised.append(
                replace(
                    region,
                    value=None,
                    symbology=None,
                    status=RegionStatus.REVIEW_CANDIDATE,
                    confidence=min(region.confidence, 0.5),
                    extras=_details(
                        region,
                        reason="short-checksum-free-itf",
                        unconfirmed_value=region.value,
                        unconfirmed_symbology=region.symbology,
                    ),
                )
            )
            continue
        replacements = [region]
        if region.status is RegionStatus.DECODED and region.kind == "linear":
            if (_joined(region) or _wide_recovery(region)) and pixels() is not None:
                replacements = _refine_recovery(region, gray)
            for index, replacement in enumerate(replacements):
                _, _, _, length, height = _axes(replacement)
                # A decoder rectangle describes successful scan rows, not
                # necessarily the full physical symbol. Reconstruct that field
                # before matching observations, even when no proposal overlaps
                # the sampled band. Its aspect ratio cannot establish whether
                # the decoder reached the top and bottom of the bars.
                if (
                    length >= 40
                    and (_observed(replacement) or height <= max(3.0, length * 0.015))
                    and pixels() is not None
                ):
                    replacements[index] = _extend_scanline(
                        replacement, gray, partial_field=_observed(replacement)
                    )
            geometry_refinements += sum(r.quad != region.quad for r in replacements)
        revised.extend(replacements)

    def suppress(region: Region, reason: str) -> None:
        suppressed.append(
            {
                "reason": reason,
                "value": region.value,
                "symbology": region.symbology,
                "status": region.status.value,
                "quad": list(region.quad),
                "sources": list(region.sources),
                "evidence": region.extras.get("reconciliation", {}),
            }
        )

    decoded = []
    duplicate_proposals: list[list[Region]] = []
    for candidate in sorted(
        (r for r in revised if r.status is RegionStatus.DECODED), key=_rank
    ):
        duplicate = next(
            (
                index
                for index, prior in enumerate(decoded)
                if _same_symbol(prior, candidate, direction)
            ),
            None,
        )
        if duplicate is not None:
            duplicate_proposals[duplicate].append(candidate)
            suppress(candidate, "duplicate-decoded-symbol")
        else:
            decoded.append(candidate)
            duplicate_proposals.append([])
    physical = [
        _physical_geometry(region, proposals, direction)
        for region, proposals in zip(decoded, duplicate_proposals, strict=True)
    ]
    geometry_refinements += sum(
        before.quad != after.quad
        for before, after in zip(decoded, physical, strict=True)
    )
    decoded = physical
    rank = {
        RegionStatus.UNRESOLVED_MATRIX: 3,
        RegionStatus.UNRESOLVED_LINEAR: 3,
        RegionStatus.REVIEW_CANDIDATE: 2,
        RegionStatus.LOCALIZED: 1,
    }
    undecoded = []
    for candidate in sorted(
        (r for r in revised if r.status is not RegionStatus.DECODED),
        key=lambda r: (-rank.get(r.status, 0), *_rank(r)),
    ):
        if _covered(candidate, decoded):
            suppress(candidate, "covered-by-decoded-symbol")
        elif any(
            prior.kind == candidate.kind
            and _aligned(prior, candidate)
            and _overlap(prior, candidate) >= 0.65
            for prior in undecoded
        ):
            suppress(candidate, "duplicate-undecoded-symbol")
        else:
            undecoded.append(candidate)
    # Geometry resolution can connect two disjoint decoder strips to one
    # independently confirmed bar field. Recheck the resolved footprints; do not
    # use transitive raw proposal overlap (a joined parent can span real copies).
    consolidated: list[Region] = []
    for candidate in sorted(
        decoded,
        key=lambda r: (
            r.extras.get("reconciliation", {}).get("geometry")
            not in {
                "confirmed-localizer-field",
                "supported-bar-field",
            },
            *_rank(r),
        ),
    ):
        if any(_same_symbol(prior, candidate, direction) for prior in consolidated):
            suppress(candidate, "duplicate-after-geometry-refinement")
        else:
            consolidated.append(candidate)
    resolved, conflicting_symbols = _resolve_payload_conflicts(consolidated, direction)
    output = sorted(
        [*resolved, *undecoded],
        key=lambda r: (min(r.quad[1::2]), min(r.quad[0::2]), r.value or ""),
    )
    return output, {
        "input_regions": len(candidates),
        "output_regions": len(output),
        "deduplicated_or_covered": len(suppressed),
        "geometry_refinements": geometry_refinements,
        "short_itf_reviews": short_reads,
        "conflicting_symbols": conflicting_symbols,
        "suppressed": suppressed,
    }
