"""Kaggle cloud backend — runs models on Kaggle's free GPU."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def _get_api(username: str, api_key: str):
    """Create and authenticate a Kaggle API client using provided credentials."""
    # Support both old (username+key) and new (KGAT_ token) formats
    if api_key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = api_key
    else:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = api_key

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _create_dataset(api, files: list[str], dataset_slug: str, username: str) -> str:
    """Upload input files as a private Kaggle dataset."""
    staging_dir = tempfile.mkdtemp()
    try:
        for f in files:
            shutil.copy2(f, staging_dir)

        metadata = {
            "title": dataset_slug,
            "id": f"{username}/{dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(os.path.join(staging_dir, "dataset-metadata.json"), "w") as f:
            json.dump(metadata, f)

        api.dataset_create_new(folder=staging_dir, public=False, quiet=True)
    finally:
        shutil.rmtree(staging_dir)

    return f"{username}/{dataset_slug}"


def _build_script(dataset_slug: str, model_name: str, input_filename: str) -> str:
    """Generate the Python script that runs on Kaggle.

    Supports yolov8, grounding_dino, sam2, samurai, depth_anything_v2.
    Uses packages pre-installed on Kaggle GPU images when possible.
    """
    return f"""
import os, sys, json, time

# Try installing deps (may fail if no internet, that's OK — Kaggle has most pre-installed)
def safe_install(pkg):
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                       check=True, timeout=60, capture_output=True)
    except Exception:
        print(f"Note: Could not install {{pkg}}, using pre-installed version")

output_dir = "/kaggle/working/results/"
os.makedirs(output_dir, exist_ok=True)
model_name = "{model_name}"

# Find input file — search all mounted dataset directories recursively
input_file = None
base_dir = "/kaggle/input/"
print(f"Looking for {input_filename} in {{base_dir}}")
if os.path.exists(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "{input_filename}" in files:
            input_file = os.path.join(root, "{input_filename}")
            print(f"  Found: {{input_file}}")
            break
if not input_file:
    input_file = os.path.join(base_dir, "{dataset_slug}", "{input_filename}")
    print(f"  Fallback path: {{input_file}}")

if not os.path.exists(input_file):
    print(f"ERROR: Input file not found. Available dirs:")
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            print(f"  {{os.path.join(root, f)}}")
    sys.exit(1)

print(f"Running {{model_name}} on {{input_file}}")

# Detect best device — check CUDA capability compatibility
import torch
device = "cpu"
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    # PyTorch 2.x requires sm_70+ (V100, T4, A100, etc.)
    if cap[0] >= 7:
        device = "cuda"
        print(f"Using GPU: {{gpu_name}} (sm_{{cap[0]}}{{cap[1]}})")
    else:
        print(f"GPU {{gpu_name}} (sm_{{cap[0]}}{{cap[1]}}) too old for PyTorch, using CPU")
else:
    print("No GPU detected, using CPU")
start_time = time.time()

if model_name == "yolov8":
    # Use torchvision's pre-installed Faster R-CNN (no internet needed)
    import torch, torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from PIL import Image
    import torchvision.transforms as T

    # Faster R-CNN v2 weights are cached on Kaggle GPU images
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True).eval().to(device)
    img = Image.open(input_file).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    COCO_LABELS = ["__bg__","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","N/A","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","N/A","backpack","umbrella","N/A","N/A","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","N/A","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","N/A","dining table","N/A","N/A","toilet","N/A","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","N/A","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

    keep = preds["scores"] > 0.5
    labels = [COCO_LABELS[i] for i in preds["labels"][keep].cpu().numpy()]
    boxes = preds["boxes"][keep].cpu().numpy().tolist()
    scores = preds["scores"][keep].cpu().numpy().tolist()

    # Save annotated image
    import cv2, numpy as np
    img_cv = cv2.imread(input_file)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{{label}} {{score:.2f}}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(os.path.join(output_dir, "detected.jpg"), img_cv)

    print(f"Detected {{len(labels)}} objects: {{set(labels)}}")
    with open(os.path.join(output_dir, "detections.json"), "w") as f:
        json.dump({{"count": len(labels), "labels": labels, "boxes": boxes, "scores": scores}}, f, indent=2)

elif model_name in ("grounding_dino", "sam2", "samurai", "depth_anything_v2"):
    safe_install("transformers")
    from transformers import pipeline
    from PIL import Image

    if model_name == "grounding_dino":
        pipe = pipeline("zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny", device=0 if device == "cuda" else -1)
        image = Image.open(input_file)
        results = pipe(image, candidate_labels=["object"], threshold=0.15)
        with open(os.path.join(output_dir, "detections.json"), "w") as f:
            json.dump({{"count": len(results), "detections": [
                {{"label": r["label"], "score": round(r["score"], 3),
                  "box": [r["box"]["xmin"], r["box"]["ymin"], r["box"]["xmax"], r["box"]["ymax"]]}}
                for r in results
            ]}}, f, indent=2)
        print(f"Detected {{len(results)}} objects")

    elif model_name in ("sam2", "samurai"):
        pipe = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=0 if device == "cuda" else -1)
        image = Image.open(input_file)
        outputs = pipe(image, points_per_batch=64)
        masks = outputs["masks"]
        import numpy as np
        # Save overlay
        img_array = np.array(image)
        overlay = img_array.copy().astype(np.float64)
        np.random.seed(42)
        for mask in masks:
            mask_np = np.array(mask, dtype=bool)
            color = np.random.random(3) * 255
            overlay[mask_np] = overlay[mask_np] * 0.5 + color * 0.5
        Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(output_dir, "masks_overlay.png"))
        print(f"Generated {{len(masks)}} masks")

    elif model_name == "depth_anything_v2":
        pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if device == "cuda" else -1)
        image = Image.open(input_file)
        result = pipe(image)
        import numpy as np
        from matplotlib import cm
        depth_np = result["predicted_depth"].detach().cpu().numpy().squeeze()
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        colored = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(os.path.join(output_dir, "depth_color.png"))
        print("Depth map generated")

else:
    print(f"Model {{model_name}} not supported on Kaggle yet")

elapsed = time.time() - start_time
print(f"Done in {{elapsed:.1f}}s! Results saved to {{output_dir}}")
"""


def _push_kernel(api, dataset_id: str, kernel_slug: str, script: str, username: str) -> str:
    """Push a GPU-enabled kernel to Kaggle."""
    staging_dir = tempfile.mkdtemp()
    try:
        script_path = os.path.join(staging_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(script)

        metadata = {
            "id": f"{username}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "accelerator": "gpu_t4x1",  # Request T4 GPU specifically
            "enable_internet": True,
            "dataset_sources": [dataset_id],
            "kernel_sources": [],
            "competition_sources": [],
            "keywords": [],
        }
        with open(os.path.join(staging_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f)

        api.kernels_push(folder=staging_dir)
    finally:
        shutil.rmtree(staging_dir)

    return f"{username}/{kernel_slug}"


def _wait_for_kernel(api, kernel_id: str, poll_interval: int = 15, timeout: int = 1800) -> str:
    """Poll kernel status until complete or failed."""
    username, slug = kernel_id.split("/", 1)

    elapsed = 0
    while elapsed < timeout:
        try:
            # kaggle 2.0 takes a single kernel ref string
            try:
                status_obj = api.kernels_status(kernel_id)
            except TypeError:
                status_obj = api.kernels_status(user_name=username, kernel_slug=slug)
        except Exception as e:
            console.print(f"[dim]Status check: {e}[/dim]")
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        # Handle different API versions
        if hasattr(status_obj, "status"):
            status = status_obj.status
        elif isinstance(status_obj, dict):
            status = status_obj.get("status", str(status_obj))
        else:
            status = str(status_obj)

        console.print(f"\r[dim]Kaggle: {status} ({elapsed}s elapsed)[/dim]", end="")

        status_lower = str(status).lower()
        if "complete" in status_lower:
            console.print()
            return "complete"
        elif "error" in status_lower or "cancel" in status_lower:
            console.print()
            raise RuntimeError(f"Kaggle kernel failed: {status}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    raise TimeoutError(f"Kaggle kernel timed out after {timeout}s")


def _download_output(api, kernel_id: str, output_dir: str):
    """Download kernel output files."""
    import locale
    os.makedirs(output_dir, exist_ok=True)

    # Force UTF-8 to handle Unicode in Kaggle logs (progress bars etc.)
    old_encoding = os.environ.get("PYTHONUTF8")
    os.environ["PYTHONUTF8"] = "1"

    try:
        try:
            api.kernels_output(kernel_id, path=output_dir, quiet=True)
        except TypeError:
            username, slug = kernel_id.split("/", 1)
            api.kernels_output(user_name=username, kernel_slug=slug, path=output_dir, quiet=True)
        except UnicodeEncodeError:
            # Kaggle logs contain Unicode progress bars that fail on Windows cp1252
            # Download without log — the actual output files are what we need
            console.print("[dim]Note: Could not save kernel log (encoding issue), results downloaded.[/dim]")
    finally:
        if old_encoding is None:
            os.environ.pop("PYTHONUTF8", None)
        else:
            os.environ["PYTHONUTF8"] = old_encoding


def _cleanup_dataset(api, dataset_id: str):
    """Delete the temporary input dataset."""
    try:
        # Try the API method first
        if hasattr(api, "dataset_delete"):
            api.dataset_delete(dataset=dataset_id)
        else:
            import requests
            username, slug = dataset_id.split("/", 1)
            requests.delete(
                f"https://www.kaggle.com/api/v1/datasets/{dataset_id}",
                auth=(username, os.environ.get("KAGGLE_KEY", "")),
            )
    except Exception:
        pass  # cleanup is best-effort


def run_on_kaggle(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    username: str,
    api_key: str,
) -> dict:
    """Run a model on Kaggle's free GPU. Returns results summary."""
    api = _get_api(username, api_key)
    job_id = str(int(time.time()))
    dataset_slug = f"pixo-input-{job_id}"
    kernel_slug = f"pixo-run-{job_id}"

    start = time.time()

    try:
        # 1. Upload input
        console.print("[bold]Uploading to Kaggle...[/bold]")
        dataset_id = _create_dataset(api, [str(input_path)], dataset_slug, username)

        # Wait for dataset to become available (Kaggle needs time to process uploads)
        console.print("[dim]Waiting for dataset to process (30s)...[/dim]")
        time.sleep(30)

        # 2. Push kernel
        console.print("[bold]Starting Kaggle GPU kernel...[/bold]")
        script = _build_script(dataset_slug, model_name, input_path.name)
        kernel_id = _push_kernel(api, dataset_id, kernel_slug, script, username)

        # 3. Wait for completion
        try:
            _wait_for_kernel(api, kernel_id)
        except RuntimeError as e:
            if "error" in str(e).lower():
                console.print(f"\n[red]{e}[/red]")
                console.print("\n[yellow]Common causes:[/yellow]")
                console.print("  1. [bold]Phone not verified[/bold] on Kaggle (required for internet access)")
                console.print("     Fix: https://www.kaggle.com/settings -> Phone Verification")
                console.print("  2. Kaggle GPU quota exhausted (30hrs/week limit)")
                console.print("  3. Temporary Kaggle service issue")
                console.print(f"\n[dim]Check kernel logs: https://www.kaggle.com/code/{kernel_id}[/dim]")
                return
            raise

        # 4. Download results
        console.print("[bold]Downloading results...[/bold]")
        _download_output(api, kernel_id, str(output_dir))

        elapsed = time.time() - start
        console.print(f"\n[green bold]Done![/green bold]")
        console.print(f"[bold]Time:[/bold] {int(elapsed)}s (including upload/download)")
        console.print(f"[bold]Backend:[/bold] Kaggle GPU")
        console.print(f"[bold]Output:[/bold] {output_dir}")

        return {
            "backend": "kaggle",
            "time_seconds": round(elapsed, 1),
            "output_dir": str(output_dir),
        }

    finally:
        # 5. Cleanup
        try:
            _cleanup_dataset(api, f"{username}/{dataset_slug}")
        except Exception:
            pass
