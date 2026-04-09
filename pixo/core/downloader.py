"""Model downloader — downloads weights from HuggingFace to ~/.pixo/models/."""

from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console

from pixo.core.plugin import ModelCard, ModelVariant

MODELS_DIR = Path.home() / ".pixo" / "models"

console = Console()


def get_model_dir(model_name: str, variant_name: str | None = None) -> Path:
    """Return the local directory for a model variant."""
    subdir = model_name if not variant_name or variant_name == "default" else f"{model_name}_{variant_name}"
    return MODELS_DIR / subdir


def get_model_path(model_name: str, variant: ModelVariant, variant_name: str | None = None) -> Path:
    """Return the expected local path for a model file."""
    return get_model_dir(model_name, variant_name) / variant.filename


def is_downloaded(model_name: str, variant: ModelVariant, variant_name: str | None = None) -> bool:
    """Check if a model variant is already downloaded."""
    path = get_model_path(model_name, variant, variant_name)
    return path.exists()


def download_model(model: ModelCard, variant_name: str | None = None) -> Path:
    """Download a model variant from HuggingFace. Returns the local file path."""
    variant = model.get_variant(variant_name)
    model_dir = get_model_dir(model.name, variant_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    dest_path = model_dir / variant.filename

    if dest_path.exists():
        console.print(f"[green]Already downloaded:[/green] {dest_path}")
        return dest_path

    # Use per-variant repo if specified, otherwise fall back to model-level repo
    repo = variant.repo or model.huggingface_repo
    label = f"{model.name}:{variant_name}" if variant_name and variant_name != "default" else model.name
    console.print(f"[bold]Pulling {label}[/bold] ({variant.size_mb} MB) from {repo}")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo,
            filename=variant.filename,
            local_dir=str(model_dir),
        )
        console.print(f"[green]Done![/green] Saved to {dest_path}")
        return Path(downloaded_path)
    except Exception as e:
        # For Ultralytics models (YOLO, RT-DETR), the runner handles downloading
        # Create a marker so pixo knows the model dir exists
        if variant.filename.endswith(".pt"):
            console.print(f"[dim]Will download via Ultralytics on first run.[/dim]")
            return dest_path
        raise


def remove_model(model_name: str, variant_name: str | None = None) -> bool:
    """Remove a downloaded model. Returns True if something was deleted."""
    import shutil
    model_dir = get_model_dir(model_name, variant_name)
    if model_dir.exists():
        shutil.rmtree(model_dir)
        return True
    return False


def list_downloaded() -> list[str]:
    """Return names of all downloaded models."""
    if not MODELS_DIR.exists():
        return []
    return [d.name for d in MODELS_DIR.iterdir() if d.is_dir()]
