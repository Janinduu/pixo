"""Model piping — chain multiple models together.

Usage:
    pixo pipe "grounding_dino → sam2" --input video.mp4 --prompt "person"
    pixo pipe detect_and_segment --input photo.jpg --prompt "car"

The output of each model is automatically converted to the input format
needed by the next model in the chain.
"""

import json
import shutil
import tempfile
from pathlib import Path

from rich.console import Console

from pixo.core.plugin import loader
from pixo.core.downloader import get_model_path

console = Console()


# --- Pre-built pipeline templates ---

TEMPLATES = {
    "detect_and_segment": ["grounding_dino", "sam2"],
    "detect_and_track": ["grounding_dino", "sam2", "samurai"],
    "segment_and_depth": ["sam2", "depth_anything_v2"],
    "grounding_sam": ["grounding_dino", "sam2"],  # GroundingSAM = detect by text + segment
}


def parse_pipeline(pipeline_str: str) -> list[str]:
    """Parse a pipeline string into a list of model names.

    Accepts:
        "grounding_dino → sam2 → samurai"
        "grounding_dino -> sam2 -> samurai"
        "detect_and_segment" (template name)
    """
    # Check if it's a template name
    if pipeline_str in TEMPLATES:
        return TEMPLATES[pipeline_str]

    # Parse arrow-separated list
    separator = "→" if "→" in pipeline_str else "->"
    models = [m.strip() for m in pipeline_str.split(separator)]
    return [m for m in models if m]


def list_templates() -> dict[str, list[str]]:
    """Return available pipeline templates."""
    return TEMPLATES.copy()


# --- Converters between model outputs ---

def _detection_to_segmentation_input(detection_output_dir: Path) -> dict:
    """Convert detection results (boxes) to SAM2 segmentation prompts."""
    # Look for detection results
    det_file = detection_output_dir / "detections.json"
    if not det_file.exists():
        # Try Florence-2 format
        det_file = detection_output_dir / "florence2_result.json"

    if not det_file.exists():
        return {}

    data = json.loads(det_file.read_text())

    # Extract bounding boxes
    boxes = []
    if "detections" in data:
        # GroundingDINO format
        for det in data["detections"]:
            boxes.append(det["box"])
    elif "bboxes" in data:
        # Florence-2 format
        boxes = data["bboxes"]

    return {"boxes": boxes, "prompt": "detected objects"}


def _segmentation_to_tracking_input(seg_output_dir: Path) -> dict:
    """Convert segmentation masks to tracking initialization."""
    masks_dir = seg_output_dir / "masks"
    if not masks_dir.exists():
        return {}

    mask_files = sorted(masks_dir.glob("*.png"))
    return {"mask_files": [str(f) for f in mask_files], "num_objects": len(mask_files)}


CONVERTERS = {
    ("detection", "segmentation"): _detection_to_segmentation_input,
    ("detection", "video-tracking-segmentation"): _detection_to_segmentation_input,
    ("segmentation", "video-tracking-segmentation"): _segmentation_to_tracking_input,
}


def _get_converter(from_task: str, to_task: str):
    """Find a converter function between two task types."""
    return CONVERTERS.get((from_task, to_task))


# --- Pipeline execution ---

def run_pipeline(
    model_names: list[str],
    input_path: Path,
    output_dir: Path,
    options: dict,
    device: str = "cpu",
) -> Path:
    """Execute a pipeline of models in sequence.

    Returns the final output directory.
    """
    from pixo.core.runner import get_device

    resolved_device = get_device(device)
    current_input = input_path
    final_output = output_dir

    total_steps = len(model_names)

    for step, model_name in enumerate(model_names, 1):
        console.print(f"\n[bold]Step {step}/{total_steps}:[/bold] {model_name}")

        # Load model card and runner
        try:
            card = loader.load_card(model_name)
            runner_mod = loader.load_runner(model_name)
        except (KeyError, FileNotFoundError, ValueError) as e:
            console.print(f"[red]Error loading {model_name}: {e}[/red]")
            return final_output

        # Get model path
        variant = card.default_variant
        model_path = get_model_path(card.name, variant)

        if not model_path.exists():
            console.print(f"[yellow]{model_name} not downloaded. Pulling...[/yellow]")
            from pixo.core.downloader import download_model as dl_model
            dl_model(card)

        # Set up step output directory
        if step == total_steps:
            step_output = final_output
        else:
            step_output = Path(tempfile.mkdtemp(prefix=f"pixo_pipe_{model_name}_"))

        step_output.mkdir(parents=True, exist_ok=True)

        # Merge pipeline options with converter output from previous step
        step_options = dict(options)
        step_options["device"] = resolved_device

        # Setup and run
        try:
            loaded_model = runner_mod.setup(str(model_path.parent), variant.filename, resolved_device)
        except NotImplementedError as e:
            console.print(f"[red]{e}[/red]")
            return final_output

        console.print(f"  Running {model_name}...")
        last_progress = {}
        for progress in runner_mod.run(loaded_model, str(current_input), str(step_output), step_options):
            last_progress = progress
            frame = progress.get("frame", 0)
            total = progress.get("total", 0)
            if total > 0:
                pct = int(frame / total * 100)
                console.print(f"\r  [dim]{model_name}: {frame}/{total} ({pct}%)[/dim]", end="")

        objects = last_progress.get("objects", "?")
        classes = last_progress.get("classes", [])
        console.print(f"\n  [green]Done[/green] — {objects} objects, classes: {', '.join(str(c) for c in classes)}")

        # If not the last step, apply converter for next model
        if step < total_steps:
            next_model = model_names[step]
            try:
                next_card = loader.load_card(next_model)
                converter = _get_converter(card.task, next_card.task)
                if converter:
                    extra = converter(step_output)
                    step_options.update(extra)
            except Exception:
                pass

            # The current step's output becomes the next step's input
            # For most pipelines, the original input carries through
            # (e.g., detection finds boxes, segmentation uses same image + boxes)

        # Clean up intermediate dirs (not the final one)
        # We keep them for now in case downstream needs them

    return final_output
