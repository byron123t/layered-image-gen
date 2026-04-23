#!/usr/bin/env python3
"""Render an image composition JSON as an outlined Matplotlib figure."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

TYPE_COLORS = {
    "text": "#1f77b4",
    "icon": "#ff7f0e",
    "background": "#2ca02c",
    "foreground": "#d62728",
    "other": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render image composition regions from JSON into an image. "
            "Regions are drawn in array order, with increasing opacity."
        )
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to an image composition JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("image_composition_render.png"),
        help="Output image path (default: image_composition_render.png).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("W", "H"),
        default=(12.0, 7.0),
        help="Figure size in inches, e.g. --figsize 12 7.",
    )
    return parser.parse_args()


def load_composition(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_points(points: Optional[Sequence[Sequence[float]]]) -> List[Point]:
    if not points:
        return []
    out: List[Point] = []
    for pt in points:
        if len(pt) != 2:
            continue
        out.append((float(pt[0]), float(pt[1])))
    return out


def _centroid(points: Sequence[Point]) -> Optional[Point]:
    if not points:
        return None
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


def _bounds_from_points(points: Sequence[Point]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _alpha_for_region(idx: int, total: int, region_type: str) -> float:
    if total <= 1:
        base = 1.0
    else:
        min_v = np.log(1)
        max_v = np.log(total)
        v = np.log(idx + 1)
        frac = (v - min_v) / (max_v - min_v) if total > 1 else 1.0
        base = 0.2 + 0.8 * frac
    if region_type == "background":
        return min(base, 0.4)
    if region_type == "foreground":
        return 1.0
    return base


def _ellipse_sizes(region: dict, points: Sequence[Point]) -> Tuple[Optional[float], Optional[float]]:
    attrs = region.get("ellipse_attribute")
    if isinstance(attrs, dict):
        if "radius" in attrs:
            r = float(attrs["radius"])
            return 2.0 * r, 2.0 * r
        if (
            "horizontal_axis_length" in attrs
            and "vertical_axis_length" in attrs
        ):
            return float(attrs["horizontal_axis_length"]), float(attrs["vertical_axis_length"])

    bounds = _bounds_from_points(points)
    if bounds is None:
        return None, None
    min_x, min_y, max_x, max_y = bounds
    return max_x - min_x, max_y - min_y


def _shape_patch(region: dict, color: str, alpha: float) -> Optional[patches.Patch]:
    shape = region.get("region_shape")
    origin = region.get("origin") or [0.0, 0.0]
    ox = float(origin[0])
    oy = float(origin[1])
    orientation = float(region.get("orientation", 0.0))
    points = _to_points(region.get("points"))

    if shape == "other":
        return None

    if shape in {"polygon", "rectangle", "square"} and points:
        return patches.Polygon(
            points,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            alpha=alpha,
        )

    if shape == "circle":
        width, height = _ellipse_sizes(region, points)
        if width is None:
            return None
        return patches.Ellipse(
            (ox, oy),
            width=width,
            height=height,
            angle=orientation,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            alpha=alpha,
        )

    if shape == "ellipse":
        width, height = _ellipse_sizes(region, points)
        if width is None or height is None:
            return None
        return patches.Ellipse(
            (ox, oy),
            width=width,
            height=height,
            angle=orientation,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            alpha=alpha,
        )

    if shape in {"rectangle", "square"}:
        # Fallback for missing points: draw centered rectangle.
        fallback_w = 40.0
        fallback_h = 40.0 if shape == "square" else 25.0
        return patches.Rectangle(
            (ox - fallback_w / 2.0, oy - fallback_h / 2.0),
            fallback_w,
            fallback_h,
            angle=orientation,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            alpha=alpha,
        )

    return None


def _label_center(region: dict) -> Point:
    shape = region.get("region_shape")
    points = _to_points(region.get("points"))
    origin = region.get("origin") or [0.0, 0.0]
    ox = float(origin[0])
    oy = float(origin[1])

    if shape in {"circle", "ellipse"}:
        return (ox, oy)
    if points:
        center = _centroid(points)
        if center is not None:
            return center
    return (ox, oy)


def _expand_bounds(
    current: Optional[Tuple[float, float, float, float]],
    new: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    if current is None:
        return new
    return (
        min(current[0], new[0]),
        min(current[1], new[1]),
        max(current[2], new[2]),
        max(current[3], new[3]),
    )


def _region_bounds(region: dict) -> Optional[Tuple[float, float, float, float]]:
    shape = region.get("region_shape")
    origin = region.get("origin") or [0.0, 0.0]
    ox = float(origin[0])
    oy = float(origin[1])
    points = _to_points(region.get("points"))

    p_bounds = _bounds_from_points(points)
    if p_bounds is not None:
        return p_bounds

    if shape in {"circle", "ellipse"}:
        w, h = _ellipse_sizes(region, points)
        if w is not None and h is not None:
            return (ox - w / 2.0, oy - h / 2.0, ox + w / 2.0, oy + h / 2.0)

    return (ox - 20.0, oy - 20.0, ox + 20.0, oy + 20.0)


def render_composition(composition: dict, output_path: Path, figsize: Tuple[float, float]) -> None:
    regions = composition.get("regions", [])
    fig, ax = plt.subplots(figsize=figsize)
    bounds: Optional[Tuple[float, float, float, float]] = None
    total = len(regions)

    for idx, region in enumerate(regions):
        region_type = str(region.get("region_type", "other"))
        color = TYPE_COLORS.get(region_type, TYPE_COLORS["other"])
        alpha = _alpha_for_region(idx, total, region_type)

        patch = _shape_patch(region, color=color, alpha=alpha)
        if patch is not None:
            ax.add_patch(patch)

        center_x, center_y = _label_center(region)
        if region.get("region_shape") != "other":
            ax.text(
                center_x,
                center_y,
                region_type,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                alpha=alpha,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": min(0.85, max(0.25, alpha)),
                    "pad": 1.5,
                },
            )

        region_bounds = _region_bounds(region)
        if region_bounds is not None:
            bounds = _expand_bounds(bounds, region_bounds)

    if bounds is None:
        bounds = (0.0, 0.0, 100.0, 100.0)

    min_x, min_y, max_x, max_y = bounds
    dx = max(1.0, max_x - min_x)
    dy = max(1.0, max_y - min_y)
    pad_x = dx * 0.08
    pad_y = dy * 0.08

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_title(composition.get("title", "Image Composition"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    composition = load_composition(args.input_json)
    render_composition(composition, args.output, tuple(args.figsize))
    print(f"Rendered composition to: {args.output}")


if __name__ == "__main__":
    main()
