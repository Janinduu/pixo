"""Smart router — decides whether to run locally or on a cloud GPU."""

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pixo.core.profiler import get_profile

console = Console()


@dataclass
class RouteEstimate:
    backend: str  # "local", "kaggle", "colab"
    estimated_seconds: int
    available: bool
    reason: str = ""


def _get_frame_count(input_path: Path) -> int:
    """Get actual frame count from video, or estimate from file size."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(input_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if count > 0:
            return count
    except Exception:
        pass
    # Fallback: rough estimate from file size
    size_mb = input_path.stat().st_size / (1024 * 1024)
    return max(int(size_mb * 30), 30)


def estimate_local_time(input_path: Path, has_gpu: bool, is_optimized: bool) -> int:
    """Rough estimate of local inference time in seconds."""
    suffix = input_path.suffix.lower()
    is_video = suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    if not is_video:
        return 5 if has_gpu else 15

    frames = _get_frame_count(input_path)

    if has_gpu:
        secs_per_frame = 0.05
    elif is_optimized:
        secs_per_frame = 0.2
    else:
        secs_per_frame = 0.35

    return int(frames * secs_per_frame)


def estimate_cloud_time(input_path: Path) -> int:
    """Estimate cloud inference time (GPU + upload/download overhead)."""
    size_mb = input_path.stat().st_size / (1024 * 1024)

    # Upload/download: ~5 MB/s
    transfer_secs = int(size_mb / 5) * 2  # upload + download

    # Kaggle overhead: ~90s for kernel startup + pip install
    startup = 90

    # GPU inference on T4: ~0.02s/frame (very fast)
    suffix = input_path.suffix.lower()
    is_video = suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")
    if is_video:
        frames = _get_frame_count(input_path)
        inference = int(frames * 0.02)
    else:
        inference = 3

    return startup + transfer_secs + inference


def pick_backend(
    input_path: Path,
    model_name: str,
    is_optimized: bool,
    kaggle_configured: bool,
    colab_configured: bool,
    force_backend: str | None = None,
) -> str:
    """Decide where to run. Returns 'local', 'kaggle', or 'colab'."""

    if force_backend:
        return force_backend

    profile = get_profile()

    estimates = []

    # Local estimate
    local_secs = estimate_local_time(input_path, profile.has_gpu, is_optimized)
    estimates.append(RouteEstimate(
        backend="local",
        estimated_seconds=local_secs,
        available=True,
    ))

    # Kaggle estimate
    if kaggle_configured:
        cloud_secs = estimate_cloud_time(input_path)
        estimates.append(RouteEstimate(
            backend="kaggle",
            estimated_seconds=cloud_secs,
            available=True,
        ))
    else:
        estimates.append(RouteEstimate(
            backend="kaggle",
            estimated_seconds=0,
            available=False,
            reason="Not configured",
        ))

    # Colab estimate
    if colab_configured:
        cloud_secs = estimate_cloud_time(input_path)
        estimates.append(RouteEstimate(
            backend="colab",
            estimated_seconds=cloud_secs + 30,  # extra manual overhead
            available=True,
        ))
    else:
        estimates.append(RouteEstimate(
            backend="colab",
            estimated_seconds=0,
            available=False,
            reason="Not configured",
        ))

    # Show estimates
    table = Table(title="Estimated Times")
    table.add_column("Backend", style="bold cyan")
    table.add_column("Est. Time", justify="right")
    table.add_column("Status")

    for est in estimates:
        if est.available:
            mins, secs = divmod(est.estimated_seconds, 60)
            time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            table.add_row(est.backend, time_str, "[green]Available[/green]")
        else:
            table.add_row(est.backend, "—", f"[dim]{est.reason}[/dim]")

    console.print(table)

    # Pick the fastest available
    available = [e for e in estimates if e.available]
    best = min(available, key=lambda e: e.estimated_seconds)

    # Only suggest cloud if it saves significant time (> 2 minutes faster)
    if best.backend != "local":
        local_est = next(e for e in estimates if e.backend == "local")
        savings = local_est.estimated_seconds - best.estimated_seconds
        if savings < 120:
            # Not worth the cloud overhead
            console.print(f"\n[dim]Local is fast enough, running locally.[/dim]")
            return "local"

    console.print(f"\n[bold]Recommended:[/bold] {best.backend}")
    return best.backend
