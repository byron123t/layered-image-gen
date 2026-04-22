import base64
import os
from io import BytesIO

from openai_api import (
    image_generation,
    image_edit,
    IMAGE_GEN_MODEL,
)


def run_image_generation(prompt, save_path):
    """Generate an image from a text prompt and save it."""
    resp = image_generation(
        prompt=prompt,
        size="1024x1024",
        quality="medium",
        output_format="png",
        n=1,
    )
    b64 = resp["data"][0]["b64_json"]
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"{IMAGE_GEN_MODEL} generated: {os.path.abspath(save_path)}")


def run_image_edit(image_path, prompt, save_path, mask_path=None):
    """Edit an image given an input image and optional mask."""
    resp = image_edit(
        image_path=image_path,
        prompt=prompt,
        mask_path=mask_path,
    )
    b64 = resp["data"][0]["b64_json"]
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(b64))
    suffix = " (with mask)" if mask_path else ""
    print(f"{IMAGE_GEN_MODEL} edited{suffix}: {os.path.abspath(save_path)}")


def add_alpha_to_mask(img_path_mask, output_path):
    """Load a black & white mask, add an alpha channel, and save as PNG.

    The mask is loaded as grayscale (L), converted to RGBA, and the grayscale
    values are written into the alpha channel so white = opaque, black = transparent.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required. Install it with `pip install Pillow`."
        ) from exc

    mask = Image.open(img_path_mask).convert("L")
    mask_rgba = mask.convert("RGBA")
    mask_rgba.putalpha(mask)

    buf = BytesIO()
    mask_rgba.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(mask_bytes)


def generate_full_image_mask(image_path, mask_path):
    """Generate a mask that covers the entire image with an alpha channel.

    The mask is a solid white PNG (same size as the input image), with the
    alpha channel set from the luminance so it is fully opaque.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to generate a full-image mask. "
            "Install it with `pip install Pillow`."
        ) from exc

    img = Image.open(image_path)
    size = img.size
    # 1. Solid white grayscale mask
    mask = Image.new("L", size, 255)
    # 2. RGBA with alpha channel filled from the mask
    mask_rgba = mask.convert("RGBA")
    mask_rgba.putalpha(mask)

    os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
    mask_rgba.save(mask_path)


if __name__ == "__main__":
    # Text-to-image generation
    os.makedirs("static", exist_ok=True)
    # run_image_generation(
    #     prompt="A photograph of a red fox in an autumn forest",
    #     save_path="static/generated_image.png",
    # )

    #generate_full_image_mask(
    #    image_path="static/input_image.png",
    #    mask_path="static/mask_2.png",
    #)

    # Image edit using input_image.jpg without a mask
    input_image = "image.png"
    if os.path.isfile(input_image):
        run_image_edit(
            image_path=input_image,
            prompt="Generate an image of a fox."
            save_path="image.png",
        )
    else:
        print(f"Skipping edit: {input_image} not found.")
