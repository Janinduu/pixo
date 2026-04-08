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
    """Generate the Python script that runs on Kaggle."""
    return f"""
import subprocess, os, glob, sys

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"], check=True)

from ultralytics import YOLO

# Find input file
input_dir = "/kaggle/input/{dataset_slug}/"
input_file = os.path.join(input_dir, "{input_filename}")
output_dir = "/kaggle/working/"

print(f"Running {model_name} on {{input_file}}")
print(f"Files in input dir: {{os.listdir(input_dir)}}")

# Download and run model
model = YOLO("yolov8n.pt")
results = model.predict(
    source=input_file,
    save=True,
    project=output_dir,
    name="results",
    exist_ok=True,
    verbose=True,
)

# Print summary
if hasattr(results, '__len__'):
    total = sum(len(r.boxes) for r in results)
    print(f"Total objects detected: {{total}}")
print("Done! Results saved to /kaggle/working/results/")
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
    username, slug = kernel_id.split("/", 1)
    os.makedirs(output_dir, exist_ok=True)
    api.kernels_output(user_name=username, kernel_slug=slug, path=output_dir, quiet=True)


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

        # Wait for dataset to become available
        console.print("[dim]Waiting for dataset to process...[/dim]")
        time.sleep(15)

        # 2. Push kernel
        console.print("[bold]Starting Kaggle GPU kernel...[/bold]")
        script = _build_script(dataset_slug, model_name, input_path.name)
        kernel_id = _push_kernel(api, dataset_id, kernel_slug, script, username)

        # 3. Wait for completion
        _wait_for_kernel(api, kernel_id)

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
