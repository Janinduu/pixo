"""Depth Anything V2 runner -- monocular depth estimation via transformers."""

from pathlib import Path

VARIANT_REPOS = {
    "default": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def setup(model_dir: str, variant: str, device: str):
    """Load depth estimation pipeline."""
    from transformers import pipeline

    # Pick repo based on variant directory name
    repo = VARIANT_REPOS["default"]
    model_dir_name = Path(model_dir).name
    for key, repo_id in VARIANT_REPOS.items():
        if key in model_dir_name:
            repo = repo_id
            break

    dev = 0 if device == "cuda" else -1
    return pipeline("depth-estimation", model=repo, device=dev)


def run(model, input_path: str, output_dir: str, options: dict):
    """Run depth estimation. Yields progress dicts."""
    import numpy as np
    from PIL import Image

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    is_video = suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    if is_video:
        yield from _run_video(model, input_path, output_dir)
    else:
        yield from _run_image(model, input_path, output_dir)


def _run_image(model, input_path, output_dir):
    """Process a single image."""
    import numpy as np
    from PIL import Image
    from matplotlib import cm

    image = Image.open(input_path).convert("RGB")

    yield {"frame": 0, "total": 1, "status": "estimating depth"}

    result = model(image)
    depth_tensor = result["predicted_depth"]
    depth_np = depth_tensor.detach().cpu().numpy().squeeze()

    # Normalize to 0-1
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

    # Save colored depth (inferno colormap)
    colored = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(output_dir / f"{input_path.stem}_depth_color.png")

    # Save grayscale
    gray = (depth_norm * 255).astype(np.uint8)
    Image.fromarray(gray).save(output_dir / f"{input_path.stem}_depth_gray.png")

    # Save raw numpy
    np.save(output_dir / f"{input_path.stem}_depth.npy", depth_np)

    yield {"frame": 1, "total": 1, "objects": 1, "classes": ["depth_map"]}


def _run_video(model, input_path, output_dir):
    """Process video frame by frame."""
    import cv2
    import numpy as np
    from PIL import Image
    from matplotlib import cm

    cap = cv2.VideoCapture(str(input_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{input_path.stem}_depth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = model(pil_img)
        depth_np = result["predicted_depth"].detach().cpu().numpy().squeeze()
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

        colored = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        colored_resized = cv2.resize(colored, (w, h))
        out.write(cv2.cvtColor(colored_resized, cv2.COLOR_RGB2BGR))

        yield {"frame": i + 1, "total": total, "objects": i + 1, "classes": ["depth_map"]}

    cap.release()
    out.release()
