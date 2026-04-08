"""Model optimizer — converts PyTorch models to ONNX for faster inference."""

from pathlib import Path

from rich.console import Console

console = Console()


def get_optimized_path(model_path: Path) -> Path:
    """Return the expected ONNX path for a given PyTorch model."""
    return model_path.with_suffix(".onnx")


def is_optimized(model_path: Path) -> bool:
    """Check if an ONNX-optimized version exists."""
    return get_optimized_path(model_path).exists()


def optimize_yolo(model_path: Path) -> Path:
    """Export YOLOv8 model to ONNX format. Returns the ONNX file path."""
    from ultralytics import YOLO

    onnx_path = get_optimized_path(model_path)

    if onnx_path.exists():
        console.print(f"[green]Already optimized:[/green] {onnx_path}")
        return onnx_path

    console.print(f"[bold]Optimizing {model_path.name} -> ONNX...[/bold]")

    model = YOLO(str(model_path))
    export_path = model.export(format="onnx", simplify=True)

    # ultralytics exports next to the source file, move if needed
    exported = Path(export_path)
    if exported != onnx_path:
        exported.rename(onnx_path)

    original_mb = model_path.stat().st_size / (1024 * 1024)
    optimized_mb = onnx_path.stat().st_size / (1024 * 1024)

    console.print(
        f"[green]Done![/green] {original_mb:.1f} MB -> {optimized_mb:.1f} MB "
        f"({optimized_mb / original_mb * 100:.0f}% of original)"
    )
    return onnx_path


# Map model names to their optimization functions
OPTIMIZERS = {
    "yolov8": optimize_yolo,
}


def optimize_model(model_name: str, model_path: Path) -> Path:
    """Optimize a model. Returns the optimized file path."""
    if model_name not in OPTIMIZERS:
        raise ValueError(f"No optimizer available for '{model_name}' yet.")
    return OPTIMIZERS[model_name](model_path)
