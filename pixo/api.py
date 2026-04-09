"""pixo Python SDK — programmatic API for running CV models.

Usage:
    import pixo

    # List available models
    models = pixo.list_models()

    # Pull a model
    pixo.pull("yolov8")

    # Run inference
    result = pixo.run("yolov8", input="photo.jpg")
    print(result.objects)       # 12
    print(result.classes)       # ['person', 'car', 'chair']
    print(result.output_dir)    # './pixo_output/abc123_yolov8_photo'
    print(result.time_seconds)  # 7.2

    # Run with options
    result = pixo.run("grounding_dino", input="photo.jpg", prompt="person, car")
    result = pixo.run("sam2", input="photo.jpg", backend="kaggle")
    result = pixo.run("yolov8", input="video.mp4", low_memory=True)

    # Check hardware
    hw = pixo.doctor()
    print(hw.ram_total_gb, hw.has_gpu)

    # Model piping
    result = pixo.pipe(["yolov8", "depth_anything_v2"], input="photo.jpg")
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    """Result from a pixo.run() call."""
    model: str
    input_path: str
    output_dir: str
    objects: int = 0
    classes: list[str] = field(default_factory=list)
    time_seconds: float = 0
    device: str = "cpu"
    backend: str = "local"
    job_id: str = ""


@dataclass
class ModelInfo:
    """Info about an available model."""
    name: str
    description: str
    task: str
    author: str
    variants: list[str] = field(default_factory=list)
    default_size_mb: int = 0
    downloaded: bool = False


def list_models() -> list[ModelInfo]:
    """List all available models.

    Returns:
        List of ModelInfo objects with name, description, task, variants, etc.
    """
    from pixo.core.plugin import loader
    from pixo.core.downloader import is_downloaded

    models = []
    for card in loader.list_cards():
        variant_names = [k for k in card.variants.keys() if k != "default"]
        default = card.default_variant
        dl = is_downloaded(card.name, default)
        models.append(ModelInfo(
            name=card.name,
            description=card.description,
            task=card.task,
            author=card.author,
            variants=variant_names,
            default_size_mb=default.size_mb,
            downloaded=dl,
        ))
    return models


def pull(model_name: str, variant: str = None) -> Path:
    """Download a model.

    Args:
        model_name: Model to download (e.g. "yolov8", "sam2")
        variant: Optional variant (e.g. "tiny", "small")

    Returns:
        Path to the downloaded model directory.
    """
    from pixo.core.plugin import loader
    from pixo.core.downloader import download_model

    card = loader.load_card(model_name)
    return download_model(card, variant)


def run(
    model_name: str,
    input: str,
    output: str = "./pixo_output",
    device: str = None,
    backend: str = None,
    prompt: str = None,
    task: str = None,
    low_memory: bool = False,
    background: bool = False,
    force: bool = False,
) -> RunResult:
    """Run a model on an image or video.

    Args:
        model_name: Model to run (e.g. "yolov8", "grounding_dino")
        input: Path to input image or video file
        output: Output directory (default: ./pixo_output)
        device: Force device: "cpu" or "cuda" (default: auto-detect)
        backend: Force backend: "local", "kaggle", or "colab" (default: auto)
        prompt: Text prompt for detection models like grounding_dino
        task: Task for multi-task models like florence2 (caption/detect/ocr)
        low_memory: Frame-by-frame processing with aggressive GC
        background: Run at lowest OS priority
        force: Skip resource safety check

    Returns:
        RunResult with objects detected, classes, timing, output path, etc.
    """
    import time
    from pixo.core.plugin import loader
    from pixo.core.downloader import download_model, get_model_path
    from pixo.core.runner import get_device
    from pixo.core.output import OutputFormatter

    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    card = loader.load_card(model_name)
    variant = card.default_variant
    model_path = get_model_path(card.name, variant)

    # Auto-pull if needed
    if not model_path.exists():
        download_model(card)

    # Cloud backend
    if backend in ("kaggle", "colab"):
        if backend == "kaggle":
            from pixo.cloud.config import load_config
            from pixo.cloud.kaggle_backend import run_on_kaggle
            config = load_config()
            if not config.kaggle.is_configured:
                raise RuntimeError("Kaggle not configured. Run: pixo setup-cloud --kaggle")
            start = time.time()
            run_on_kaggle(input_path, Path(output), model_name,
                          config.kaggle.username, config.kaggle.api_key)
            return RunResult(
                model=model_name, input_path=str(input_path),
                output_dir=output, time_seconds=round(time.time() - start, 1),
                backend="kaggle",
            )

    # Local execution
    resolved_device = get_device(device)
    runner_mod = loader.load_runner(model_name)
    model_dir = str(model_path.parent)

    loaded_model = runner_mod.setup(model_dir, variant.filename, resolved_device)

    # Set up output
    formatter = OutputFormatter(
        base_dir=output, model=model_name, input_name=input_path.name,
    )
    run_output_dir = str(formatter.viz_dir)

    # Build options
    options = {"device": resolved_device}
    if prompt:
        options["prompt"] = prompt
    if task:
        options["task"] = task

    # Run
    start = time.time()
    last_update = {}
    for update in runner_mod.run(loaded_model, str(input_path), run_output_dir, options):
        last_update = update

    elapsed = round(time.time() - start, 1)

    # Save standard output
    formatter.set_input_info(str(input_path), last_update.get("total", 1))
    formatter.set_timing(elapsed)
    formatter._results.device = resolved_device
    formatter._results.variant = variant.filename
    formatter.set_summary(**last_update)
    formatter.save_results_json()
    formatter.save_summary_txt()

    return RunResult(
        model=model_name,
        input_path=str(input_path),
        output_dir=str(formatter.root),
        objects=last_update.get("objects", 0),
        classes=last_update.get("classes", []),
        time_seconds=elapsed,
        device=resolved_device,
        backend="local",
        job_id=formatter.job_id,
    )


def doctor() -> dict:
    """Check hardware and return system profile.

    Returns:
        Dict with cpu_name, cpu_cores, ram_total_gb, ram_available_gb,
        has_gpu, gpu_name, gpu_vram_gb, os_name, disk_free_gb, recommendation.
    """
    from pixo.core.profiler import get_profile
    p = get_profile()
    return {
        "cpu_name": p.cpu_name,
        "cpu_cores": p.cpu_cores,
        "ram_total_gb": p.ram_total_gb,
        "ram_available_gb": p.ram_available_gb,
        "has_gpu": p.has_gpu,
        "gpu_name": p.gpu_name,
        "gpu_vram_gb": p.gpu_vram_gb,
        "os_name": p.os_name,
        "disk_free_gb": p.disk_free_gb,
        "recommendation": p.recommendation,
    }


def pipe(
    models: list[str],
    input: str,
    output: str = "./pixo_output",
    prompt: str = None,
    device: str = None,
) -> RunResult:
    """Run a pipeline of models in sequence.

    Args:
        models: List of model names (e.g. ["grounding_dino", "sam2"])
        input: Path to input image or video
        output: Output directory
        prompt: Text prompt for detection models
        device: Force device

    Returns:
        RunResult from the final model in the pipeline.
    """
    import time
    from pixo.core.pipeline import run_pipeline

    input_path = Path(input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    options = {}
    if prompt:
        options["prompt"] = prompt

    start = time.time()
    run_pipeline(models, input_path, Path(output), options, device=device or "cpu")
    elapsed = round(time.time() - start, 1)

    return RunResult(
        model=" -> ".join(models),
        input_path=str(input_path),
        output_dir=output,
        time_seconds=elapsed,
        device=device or "cpu",
        backend="local",
    )