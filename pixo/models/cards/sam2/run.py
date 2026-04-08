"""SAM2 runner -- automatic mask generation using transformers pipeline."""

from pathlib import Path

VARIANT_TO_REPO = {
    "model.safetensors": "facebook/sam2.1-hiera-large",  # default
}

REPO_MAP = {
    "large": "facebook/sam2.1-hiera-large",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "small": "facebook/sam2.1-hiera-small",
    "tiny": "facebook/sam2.1-hiera-tiny",
}


def setup(model_dir: str, variant: str, device: str):
    """Load SAM2 mask generation pipeline."""
    from transformers import pipeline

    # Pick the right repo based on variant file / directory name
    repo = "facebook/sam2.1-hiera-large"
    model_dir_name = Path(model_dir).name
    for key, repo_id in REPO_MAP.items():
        if key in model_dir_name:
            repo = repo_id
            break

    dev = 0 if device == "cuda" else -1
    return pipeline("mask-generation", model=repo, device=dev)


def run(model, input_path: str, output_dir: str, options: dict):
    """Run automatic mask generation. Yields progress dicts."""
    import numpy as np
    from PIL import Image

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    img_array = np.array(image)

    yield {"frame": 0, "total": 1, "status": "generating masks"}

    outputs = model(image, points_per_batch=64)
    masks = outputs["masks"]
    scores = outputs["scores"]

    # Save overlay
    overlay = img_array.copy().astype(np.float64)
    np.random.seed(42)
    for mask in masks:
        mask_np = np.array(mask, dtype=bool)
        color = np.random.random(3) * 255
        overlay[mask_np] = overlay[mask_np] * 0.5 + color * 0.5

    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    overlay_img.save(output_dir / f"{input_path.stem}_masks.png")

    # Save individual masks
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    for i, mask in enumerate(masks):
        mask_np = (np.array(mask, dtype=bool) * 255).astype(np.uint8)
        Image.fromarray(mask_np).save(masks_dir / f"mask_{i:03d}.png")

    # Save metadata
    import json
    meta = {
        "num_masks": len(masks),
        "scores": [float(s) for s in scores],
        "input": str(input_path),
    }
    (output_dir / "results.json").write_text(json.dumps(meta, indent=2))

    yield {"frame": 1, "total": 1, "objects": len(masks), "classes": ["segment"]}
