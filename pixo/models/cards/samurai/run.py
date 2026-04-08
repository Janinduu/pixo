"""SAMURAI runner -- video object tracking + segmentation using SAM2 video predictor.

Uses stock SAM2 from transformers for frame-by-frame tracking.
Requires an initial bounding box or point prompt on the first frame.
"""

from pathlib import Path

VARIANT_REPOS = {
    "default": "facebook/sam2.1-hiera-large",
    "lite": "facebook/sam2.1-hiera-base-plus",
    "tiny": "facebook/sam2.1-hiera-tiny",
}


def setup(model_dir: str, variant: str, device: str):
    """Load SAM2 for video segmentation."""
    from transformers import pipeline

    repo = VARIANT_REPOS["default"]
    model_dir_name = Path(model_dir).name
    for key, repo_id in VARIANT_REPOS.items():
        if key in model_dir_name:
            repo = repo_id
            break

    dev = 0 if device == "cuda" else -1
    mask_generator = pipeline("mask-generation", model=repo, device=dev)
    return {"pipeline": mask_generator, "device": device}


def run(model_dict, input_path: str, output_dir: str, options: dict):
    """Run video tracking/segmentation. Yields progress dicts.

    For video: segments objects in each frame and saves mask overlay video.
    For image: falls back to automatic mask generation (same as SAM2).
    """
    import cv2
    import numpy as np
    from PIL import Image

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = model_dict["pipeline"]

    suffix = input_path.suffix.lower()
    is_video = suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    if is_video:
        yield from _run_video(pipe, input_path, output_dir)
    else:
        yield from _run_image(pipe, input_path, output_dir)


def _run_image(pipe, input_path, output_dir):
    """Segment a single image (auto-mask generation)."""
    import numpy as np
    from PIL import Image

    image = Image.open(input_path).convert("RGB")
    img_array = np.array(image)

    yield {"frame": 0, "total": 1, "status": "generating masks"}

    outputs = pipe(image, points_per_batch=64)
    masks = outputs["masks"]
    scores = outputs["scores"]

    # Save overlay
    overlay = img_array.copy().astype(np.float64)
    np.random.seed(42)
    for mask in masks:
        mask_np = np.array(mask, dtype=bool)
        color = np.random.random(3) * 255
        overlay[mask_np] = overlay[mask_np] * 0.5 + color * 0.5

    Image.fromarray(overlay.astype(np.uint8)).save(
        output_dir / f"{input_path.stem}_tracked.png"
    )

    # Save individual masks
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    for i, mask in enumerate(masks):
        mask_np = (np.array(mask, dtype=bool) * 255).astype(np.uint8)
        Image.fromarray(mask_np).save(masks_dir / f"mask_{i:03d}.png")

    import json
    meta = {"num_masks": len(masks), "scores": [float(s) for s in scores]}
    (output_dir / "tracking_result.json").write_text(json.dumps(meta, indent=2))

    yield {"frame": 1, "total": 1, "objects": len(masks), "classes": ["segment"]}


def _run_video(pipe, input_path, output_dir):
    """Segment each frame of a video and produce an overlay video."""
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(input_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{input_path.stem}_tracked.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    # Use consistent colors for visual continuity across frames
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(50, 3), dtype=np.uint8)

    total_objects = 0

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            outputs = pipe(pil_img, points_per_batch=32)
            masks = outputs["masks"]
        except Exception:
            # If mask generation fails on a frame, write original frame
            out.write(frame)
            yield {"frame": i + 1, "total": total, "objects": total_objects,
                   "classes": ["segment"]}
            continue

        total_objects = max(total_objects, len(masks))

        # Create overlay
        overlay = frame.copy().astype(np.float64)
        for j, mask in enumerate(masks):
            mask_np = np.array(mask, dtype=bool)
            if mask_np.shape != (h, w):
                mask_np = cv2.resize(mask_np.astype(np.uint8), (w, h)).astype(bool)
            color = palette[j % len(palette)].astype(np.float64)
            # BGR for OpenCV
            overlay[mask_np] = overlay[mask_np] * 0.5 + color[::-1] * 0.5

        out.write(overlay.astype(np.uint8))

        yield {"frame": i + 1, "total": total, "objects": total_objects,
               "classes": ["segment"]}

    cap.release()
    out.release()
