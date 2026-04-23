#!/usr/bin/env python3
"""Render an image composition by iteratively filling regions.

Behavior:
- Text regions: render region_description as overlaid text.
- Non-text regions: call GPT-image-2 image edit endpoint using:
  - current non-text context image as the base image
  - a region mask targeting only the current region
  - region_description as the prompt context

The non-text context intentionally omits text regions so non-text generation is
conditioned only on previously rendered non-text content.
"""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
import textwrap
from pathlib import Path
from typing import Optional, Sequence, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from sensitive import AZURE_API_KEY, azure_endpoint

IMAGE_GEN_MODEL = "gpt-image-2"
IMAGE_API_VERSION = "2024-02-01"
EDIT_API_VERSION_FALLBACKS = ("2024-02-01", "2025-04-01-preview", "2025-03-01-preview")

Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an image composition JSON by iteratively filling regions. "
            "Text regions are drawn directly; other regions are generated with GPT-image-2 edits."
        )
    )
    parser.add_argument("composition_json", type=Path, help="Path to composition JSON.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("rendered_via_composition.png"),
        help="Final output image path.",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        help="Canvas size as WxH, e.g. 1024x1024.",
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=["low", "medium", "high"],
        help="GPT-image-2 quality.",
    )
    parser.add_argument(
        "--save-steps-dir",
        type=Path,
        default=None,
        help="Optional directory to save per-step debug images.",
    )
    return parser.parse_args()


def _parse_size(size_text: str) -> Tuple[int, int]:
    try:
        w_text, h_text = size_text.lower().split("x")
        w = int(w_text)
        h = int(h_text)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Invalid --size value: {size_text}. Expected format WxH.") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid --size value: {size_text}. Width/height must be > 0.")
    return w, h


def _load_composition(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_points(points: Optional[Sequence[Sequence[float]]]) -> list[Point]:
    if not points:
        return []
    out: list[Point] = []
    for pt in points:
        if len(pt) != 2:
            continue
        out.append((float(pt[0]), float(pt[1])))
    return out


def _bounds_from_points(points: Sequence[Point]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


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


def _region_bounds(region: dict) -> Tuple[float, float, float, float]:
    shape = str(region.get("region_shape", "other"))
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


def _composition_bounds(composition: dict) -> Tuple[float, float, float, float]:
    regions = composition.get("regions", [])
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    for region in regions:
        b = _region_bounds(region)
        min_x = min(min_x, b[0])
        min_y = min(min_y, b[1])
        max_x = max(max_x, b[2])
        max_y = max(max_y, b[3])
    if min_x == float("inf"):
        return (0.0, 0.0, 1.0, 1.0)
    return (min_x, min_y, max_x, max_y)


def _map_point(
    x: float,
    y: float,
    src_bounds: Tuple[float, float, float, float],
    dst_size: Tuple[int, int],
) -> Tuple[float, float]:
    min_x, min_y, max_x, max_y = src_bounds
    dst_w, dst_h = dst_size
    src_w = max(1e-9, max_x - min_x)
    src_h = max(1e-9, max_y - min_y)
    mx = (x - min_x) * (dst_w / src_w)
    my = (y - min_y) * (dst_h / src_h)
    return (mx, my)


def _region_bbox_px(
    region: dict,
    src_bounds: Tuple[float, float, float, float],
    dst_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = _region_bounds(region)
    p1 = _map_point(min_x, min_y, src_bounds, dst_size)
    p2 = _map_point(max_x, max_y, src_bounds, dst_size)
    left = int(round(min(p1[0], p2[0])))
    top = int(round(min(p1[1], p2[1])))
    right = int(round(max(p1[0], p2[0])))
    bottom = int(round(max(p1[1], p2[1])))
    return (left, top, right, bottom)


def _draw_region_mask(
    region: dict,
    src_bounds: Tuple[float, float, float, float],
    canvas_size: Tuple[int, int],
) -> Image.Image:
    """Create RGBA mask where transparent area is editable."""
    mask = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    shape = str(region.get("region_shape", "other"))
    if shape == "other":
        return mask

    points = _to_points(region.get("points"))
    mapped_points = [
        _map_point(px, py, src_bounds, canvas_size)
        for (px, py) in points
    ]
    if shape in {"polygon", "rectangle", "square"} and mapped_points:
        draw.polygon(mapped_points, fill=(0, 0, 0, 0))
        return mask

    origin = region.get("origin") or [0.0, 0.0]
    ox, oy = _map_point(float(origin[0]), float(origin[1]), src_bounds, canvas_size)
    if shape in {"circle", "ellipse"}:
        w, h = _ellipse_sizes(region, points)
        if w is None or h is None:
            bbox = _region_bbox_px(region, src_bounds, canvas_size)
            draw.ellipse(bbox, fill=(0, 0, 0, 0))
            return mask
        # Scale axes from composition-space to canvas-space.
        min_x, min_y, max_x, max_y = src_bounds
        sx = canvas_size[0] / max(1e-9, (max_x - min_x))
        sy = canvas_size[1] / max(1e-9, (max_y - min_y))
        half_w = (w * sx) / 2.0
        half_h = (h * sy) / 2.0
        bbox = (ox - half_w, oy - half_h, ox + half_w, oy + half_h)
        draw.ellipse(bbox, fill=(0, 0, 0, 0))
        return mask

    bbox = _region_bbox_px(region, src_bounds, canvas_size)
    draw.rectangle(bbox, fill=(0, 0, 0, 0))
    return mask


def _draw_text_region(
    image: Image.Image,
    region: dict,
    src_bounds: Tuple[float, float, float, float],
    canvas_size: Tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = _region_bbox_px(region, src_bounds, canvas_size)
    left = max(0, left)
    top = max(0, top)
    right = min(canvas_size[0], right)
    bottom = min(canvas_size[1], bottom)
    if right <= left or bottom <= top:
        return

    text = str(region.get("region_description", "")).strip()
    if not text:
        return

    box_w = right - left
    box_h = bottom - top
    # Simple proportional font sizing fallback.
    font_size = max(12, min(72, int(min(box_w, box_h) * 0.18)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(text, width=max(12, box_w // max(1, font_size // 2)))
    draw.rectangle([left, top, right, bottom], fill=(255, 255, 255, 120))
    draw.multiline_text(
        (left + 8, top + 8),
        wrapped,
        fill=(0, 0, 0, 255),
        font=font,
        spacing=4,
    )


def _response_to_image(resp_json: dict) -> Image.Image:
    b64 = resp_json["data"][0]["b64_json"]
    raw = base64.b64decode(b64)
    from io import BytesIO

    return Image.open(BytesIO(raw)).convert("RGBA")


def _gpt_image_edit(
    image_path: Path,
    mask_path: Path,
    prompt: str,
    size: str,
    quality: str,
) -> dict:
    if not AZURE_API_KEY:
        raise RuntimeError("AZURE_API_KEY is not set.")
    base = azure_endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {AZURE_API_KEY}"}
    data = {
        "prompt": prompt,
        "n": "1",
        "size": size,
        "quality": quality,
    }
    errors: list[str] = []
    for api_version in EDIT_API_VERSION_FALLBACKS:
        url = (
            f"{base}/openai/deployments/{IMAGE_GEN_MODEL}/images/edits"
            f"?api-version={api_version}"
        )
        with image_path.open("rb") as img, mask_path.open("rb") as msk:
            files = {
                "image": (image_path.name, img, "image/png"),
                "mask": (mask_path.name, msk, "image/png"),
            }
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=20000)
        if resp.ok:
            return resp.json()
        errors.append(f"api-version={api_version}: {resp.status_code} {resp.reason}")
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
    raise RuntimeError(
        f"{IMAGE_GEN_MODEL} /images/edits failed.\n"
        + "\n".join(errors)
    )


def _non_text_prompt(region: dict) -> str:
    desc = str(region.get("region_description", "")).strip()
    rtype = str(region.get("region_type", "other"))
    shape = str(region.get("region_shape", "other"))
    return (
        "Edit the provided image only in the transparent mask area. "
        "Preserve all unmasked regions exactly. "
        "Do not add text, letters, words, logos, or captions. "
        f"Render a {rtype} region with shape intent '{shape}'. "
        f"Region description: {desc}"
    )


def _alpha_mask_from_rgba(mask_rgba: Image.Image) -> Image.Image:
    return mask_rgba.getchannel("A")


def render_via_composition(
    composition: dict,
    output_path: Path,
    canvas_size: Tuple[int, int],
    quality: str,
    save_steps_dir: Optional[Path],
) -> None:
    regions = composition.get("regions", [])
    src_bounds = _composition_bounds(composition)
    width, height = canvas_size

    text_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    non_text_context = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    if save_steps_dir is not None:
        save_steps_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="render_composition_") as tmpdir:
        tmp = Path(tmpdir)
        for idx, region in enumerate(regions):
            region_type = str(region.get("region_type", "other"))
            step_num = idx + 1
            print(f"Rendering region {step_num}/{len(regions)} ({region_type})", flush=True)

            if region_type == "text":
                _draw_text_region(text_overlay, region, src_bounds, canvas_size)
                if save_steps_dir is not None:
                    preview = Image.alpha_composite(non_text_context, text_overlay)
                    preview.save(save_steps_dir / f"step_{step_num:02d}_text.png")
                continue

            if str(region.get("region_shape", "other")) == "other":
                print("  Skipping region_shape=other for non-text generation.", flush=True)
                continue

            base_path = tmp / f"base_{idx}.png"
            mask_path = tmp / f"mask_{idx}.png"
            non_text_context.save(base_path)
            mask_rgba = _draw_region_mask(region, src_bounds, canvas_size)
            mask_rgba.save(mask_path)

            prompt = _non_text_prompt(region)
            resp = _gpt_image_edit(
                image_path=base_path,
                mask_path=mask_path,
                prompt=prompt,
                size=f"{width}x{height}",
                quality=quality,
            )
            edited = _response_to_image(resp).resize((width, height))
            # Restrict updates to editable region only.
            editable_alpha = _alpha_mask_from_rgba(mask_rgba)
            edited_region = Image.composite(
                edited,
                non_text_context,
                Image.eval(editable_alpha, lambda a: 255 - a),
            )
            non_text_context = edited_region
            final_preview = Image.alpha_composite(non_text_context, text_overlay)

            if save_steps_dir is not None:
                mask_rgba.save(save_steps_dir / f"step_{step_num:02d}_mask.png")
                non_text_context.save(save_steps_dir / f"step_{step_num:02d}_non_text_context.png")
                final_preview.save(save_steps_dir / f"step_{step_num:02d}_final.png")

    final_image = Image.alpha_composite(non_text_context, text_overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path)


def main() -> None:
    args = parse_args()
    composition = _load_composition(args.composition_json)
    canvas_size = _parse_size(args.size)
    render_via_composition(
        composition=composition,
        output_path=args.output,
        canvas_size=canvas_size,
        quality=args.quality,
        save_steps_dir=args.save_steps_dir,
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
