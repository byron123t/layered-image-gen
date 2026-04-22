import argparse
import base64
import os

import requests

from sensitive import AZURE_API_KEY, azure_endpoint

IMAGE_GEN_MODEL = "gpt-image-2"
IMAGE_API_VERSION = "2024-02-01"
_EDIT_API_VERSION_FALLBACKS = ("2024-02-01", "2025-04-01-preview", "2025-03-01-preview")


def _post_images_generate(args):
    base = azure_endpoint.rstrip("/")
    url = (
        f"{base}/openai/deployments/{IMAGE_GEN_MODEL}/images/generations"
        f"?api-version={IMAGE_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}",
    }
    body = {
        "prompt": args.prompt,
        "n": 1,
        "size": args.size,
        "quality": args.quality,
        "output_format": args.output_format,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _post_images_edit(args):
    base = azure_endpoint.rstrip("/")
    headers = {"Authorization": f"Bearer {AZURE_API_KEY}"}
    form = {
        "prompt": args.prompt,
        "n": "1",
        "size": args.size,
        "quality": args.quality,
    }
    image_path = args.file[0]
    errors = []
    for api_version in _EDIT_API_VERSION_FALLBACKS:
        url = (
            f"{base}/openai/deployments/{IMAGE_GEN_MODEL}/images/edits"
            f"?api-version={api_version}"
        )
        with open(image_path, "rb") as img:
            files = {"image": (os.path.basename(image_path), img, "image/png")}
            resp = requests.post(url, headers=headers, files=files, data=form, timeout=300)
        if resp.ok:
            return resp.json()
        errors.append(f"  api-version={api_version}: {resp.status_code} {resp.reason}")
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
    raise RuntimeError(
        f"{IMAGE_GEN_MODEL} /images/edits is not available on this Azure resource.\n"
        + "\n".join(errors)
        + "\nVerify the deployment supports image editing in the Azure portal."
    )


def run_image_generation(args):
    if not AZURE_API_KEY:
        raise RuntimeError("AZURE_API_KEY is not set.")
    for f in args.file:
        if not os.path.isfile(f):
            raise FileNotFoundError(f)

    if args.file:
        response_json = _post_images_edit(args)
    else:
        response_json = _post_images_generate(args)

    b64 = response_json["data"][0]["b64_json"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(base64.b64decode(b64))
    print(os.path.abspath(args.output))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            f"Generate or edit an image with {IMAGE_GEN_MODEL} via the Azure images API.\n"
            "With no --file: calls /images/generations. With --file: calls /images/edits."
        )
    )
    parser.add_argument("prompt", help="Text prompt for the image.")
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Input image for editing (uses /images/edits). Only the first file is used.",
    )
    parser.add_argument(
        "--output",
        default="static/generated_image_gpt_image_2.png",
        help="Path to save the generated image.",
    )
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--output-format", default="png", choices=["png", "jpeg", "webp"])
    return parser


if __name__ == "__main__":
    run_image_generation(build_arg_parser().parse_args())
