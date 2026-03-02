#!/usr/bin/env python3
"""
Generate a NeMo Gym JSONL dataset from the PNG images in vllm_hallucination_pages/.

Each JSONL line contains:
  - responses_create_params.input: a single user message with the page image as input_image
  - verifier_metadata: image filename and path (for reference / future evaluation)

Usage:
    python generate_inference_jsonl.py \
        --images_dir /ads_ds3/data/nemotron_parse/vllm_hallucination_pages \
        --output /ads_ds3/data/nemotron_parse/inference.jsonl \
        [--max_images N]          # limit to first N images (default: all)
        [--example]               # write only 5 images to example.jsonl for NeMo Gym smoke tests
"""
import argparse
import base64
import json
import sys
from pathlib import Path


def image_to_data_url(image_path: Path) -> str:
    """Return a base64 data URL for a PNG image."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def make_jsonl_entry(image_path: Path) -> dict:
    """Build one NeMo Gym JSONL entry for a page image."""
    data_url = image_to_data_url(image_path)
    return {
        "responses_create_params": {
            # Responses API input — a single user message with the page image.
            # The content part uses "input_image" type (Responses API format).
            # vllm_model converts this to chat completions "image_url" format automatically.
            "input": [
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": data_url,
                        }
                    ],
                }
            ],
            "max_output_tokens": 4096,
        },
        # verifier_metadata is opaque to the framework — passes through to verify()
        "verifier_metadata": {
            "image_filename": image_path.name,
            "image_path": str(image_path),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate NeMo Gym JSONL for nemotron-parse inference")
    parser.add_argument(
        "--images_dir",
        default="/ads_ds3/data/nemotron_parse/vllm_hallucination_pages",
        help="Directory containing PNG page images",
    )
    parser.add_argument(
        "--output",
        default="/ads_ds3/data/nemotron_parse/inference.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit the number of images (default: all)",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Write only 5 images (for NeMo Gym example.jsonl)",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_files = sorted(images_dir.glob("*.png"))

    if not image_files:
        print(f"No PNG files found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    limit = 5 if args.example else args.max_images
    if limit is not None:
        image_files = image_files[:limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w") as f:
        for img_path in image_files:
            entry = make_jsonl_entry(img_path)
            f.write(json.dumps(entry) + "\n")
            written += 1
            print(f"[{written}/{len(image_files)}] {img_path.name}")

    print(f"\nWrote {written} entries to {output_path}")


if __name__ == "__main__":
    main()
