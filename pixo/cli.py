"""pixo CLI — main entry point."""

import time
from pathlib import Path

import psutil
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from pixo.core.plugin import loader, ModelCard
from pixo.core.downloader import download_model, remove_model, is_downloaded, get_model_path

app = typer.Typer(
    name="pixo",
    help="Ollama for Computer Vision — run heavy CV models on any laptop.",
    no_args_is_help=True,
)
console = Console()


def _parse_model_name(model_str: str) -> tuple[str, str | None]:
    """Parse 'sam2:lite' into ('sam2', 'lite'). Plain 'sam2' returns ('sam2', None)."""
    if ":" in model_str:
        name, variant = model_str.split(":", 1)
        return name, variant
    return model_str, None


def _get_card(name: str) -> ModelCard:
    """Load a model card or exit with error."""
    try:
        return loader.load_card(name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


# --- Commands ---


@app.command()
def pull(
    model_name: str = typer.Argument(help="Model to download (e.g. yolov8, sam2:small)"),
    isolate: bool = typer.Option(False, "--isolate", help="Create an isolated venv for this model"),
):
    """Download a model from HuggingFace."""
    name, variant_name = _parse_model_name(model_name)
    card = _get_card(name)

    try:
        card.get_variant(variant_name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    download_model(card, variant_name)

    if isolate:
        from pixo.core.envmanager import create_env, install_deps
        create_env(name)
        packages = card.dependencies.get("packages", [])
        if packages:
            if not install_deps(name, packages):
                console.print("[yellow]Environment created but some deps failed. You may need to install manually.[/yellow]")


@app.command()
def run(
    model_name: str = typer.Argument(help="Model to run (e.g. yolov8, sam2)"),
    input: str = typer.Option(..., "--input", "-i", help="Input image or video file"),
    output: str = typer.Option("./pixo_output", "--output", "-o", help="Output directory"),
    device: str = typer.Option(None, "--device", "-d", help="Force device: cpu or cuda"),
    backend: str = typer.Option(None, "--backend", "-b", help="Force backend: local, kaggle, or colab"),
    force: bool = typer.Option(False, "--force", help="Skip resource safety check"),
    max_ram: int = typer.Option(None, "--max-ram", help="Max RAM percent to use (default: 70)"),
    max_cpu: int = typer.Option(None, "--max-cpu", help="Max CPU cores to use"),
    low_memory: bool = typer.Option(False, "--low-memory", help="Use minimal RAM (slower but safe for 4GB machines)"),
    background: bool = typer.Option(False, "--background", help="Run at lowest priority so you can keep working"),
    isolate: bool = typer.Option(False, "--isolate", help="Run in model's isolated venv (must pixo pull --isolate first)"),
):
    """Run inference on an image or video."""
    from pixo.core.optimizer import get_optimized_path, is_optimized
    from pixo.core.guardian import check_can_run, display_safety_check
    from pixo.core.runner import get_device
    from pixo.cloud.config import load_config as load_cloud_config
    from pixo.cloud.router import pick_backend

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]File not found: {input}[/red]")
        raise typer.Exit(1)

    name, variant_name = _parse_model_name(model_name)
    card = _get_card(name)
    variant = card.get_variant(variant_name)
    model_path = get_model_path(card.name, variant, variant_name)

    # Auto-pull if not downloaded
    if not model_path.exists():
        console.print(f"[yellow]{model_name} not found locally, pulling...[/yellow]")
        download_model(card, variant_name)

    # Resource safety check (skip for cloud backends)
    if backend not in ("kaggle", "colab") and not force:
        safety = check_can_run(name, input_path, variant.size_mb)
        if not display_safety_check(safety):
            console.print("\n[dim]Use --force to run anyway, or try the suggestions above.[/dim]")
            raise typer.Exit(1)

    optimized = is_optimized(model_path)

    # Smart routing: decide where to run
    cloud_config = load_cloud_config()
    chosen = pick_backend(
        input_path=input_path,
        model_name=name,
        is_optimized=optimized,
        kaggle_configured=cloud_config.kaggle.is_configured,
        colab_configured=cloud_config.colab.is_configured,
        force_backend=backend,
    )

    if chosen == "kaggle":
        if not cloud_config.kaggle.is_configured:
            console.print("[red]Kaggle not configured. Run: pixo setup-cloud --kaggle[/red]")
            raise typer.Exit(1)
        from pixo.cloud.kaggle_backend import run_on_kaggle
        run_on_kaggle(input_path, Path(output), name,
                      cloud_config.kaggle.username, cloud_config.kaggle.api_key)
        return

    if chosen == "colab":
        from pixo.cloud.colab_backend import run_on_colab
        run_on_colab(input_path, Path(output), name)
        return

    # --- Local execution via plugin system ---

    # Check if user wants isolated env execution
    if isolate:
        from pixo.core.envmanager import env_exists
        if not env_exists(name):
            console.print(f"[red]No isolated environment for '{name}'. Run: pixo pull {name} --isolate[/red]")
            raise typer.Exit(1)

    from pixo.core.guardian import ResourceLimiter, apply_background_mode, suggest_modes, apply_low_memory_cleanup
    from pixo.core.checkpoint import CheckpointManager

    if background:
        apply_background_mode()
    if low_memory:
        console.print("[dim]Low-memory mode: processing frame-by-frame with aggressive cleanup.[/dim]")

    if not low_memory and not background:
        for suggestion in suggest_modes():
            console.print(f"[yellow]Tip:[/yellow] {suggestion}")

    ram_cap = 50 if background else (max_ram or 70)
    cpu_cap = max_cpu
    if background and cpu_cap is None:
        total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
        cpu_cap = max(total_cores - 2, 1)

    limiter = ResourceLimiter(max_ram_percent=ram_cap, max_cpu_cores=cpu_cap)

    resolved_device = get_device(device)

    # Check for existing checkpoint
    ckpt_mgr = CheckpointManager()
    existing = ckpt_mgr.find_checkpoint(name, str(input_path))
    resume_from = 0
    if existing and existing.last_frame > 0:
        console.print(f"\n[bold]Found checkpoint:[/bold] {existing.last_frame}/{existing.total_frames} "
                      f"frames ({existing.progress_percent}%)")
        if typer.confirm("Resume from checkpoint?", default=True):
            resume_from = existing.last_frame
            console.print(f"[dim]Resuming from frame {resume_from + 1}...[/dim]")

    # Load the runner via plugin system
    try:
        runner_mod = loader.load_runner(name)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    model_dir = str(model_path.parent)

    console.print(f"[bold]Device:[/bold] {resolved_device}")
    console.print(f"[bold]Input:[/bold] {input_path}")

    try:
        loaded_model = runner_mod.setup(model_dir, variant.filename, resolved_device)
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Create or reuse job state
    job = existing if existing else ckpt_mgr.create_job(
        model=name, variant=variant.filename, input_path=str(input_path),
        output_dir=str(Path(output)), device=resolved_device,
    )
    job.status = "running"

    # Get checkpoint interval from model card
    ckpt_every = card.checkpoint.every if card.checkpoint.supported else 500

    _run_with_checkpoints(
        runner_mod=runner_mod,
        loaded_model=loaded_model,
        name=name,
        input_path=input_path,
        output=output,
        resolved_device=resolved_device,
        limiter=limiter,
        low_memory=low_memory,
        job=job,
        ckpt_mgr=ckpt_mgr,
        ckpt_every=ckpt_every,
        resume_from=resume_from,
    )


def _run_with_checkpoints(runner_mod, loaded_model, name, input_path, output,
                          resolved_device, limiter, low_memory, job, ckpt_mgr,
                          ckpt_every, resume_from):
    """Execute model with checkpoint saving and Ctrl+C pause support."""
    import signal
    from pixo.core.guardian import apply_low_memory_cleanup
    from pixo.core.output import OutputFormatter

    # --- Signal handling: first Ctrl+C pauses, second kills ---
    pause_requested = False
    ctrl_c_count = 0
    last_ctrl_c = 0.0

    original_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame_obj):
        nonlocal pause_requested, ctrl_c_count, last_ctrl_c
        now = time.time()
        ctrl_c_count += 1

        if ctrl_c_count >= 2 and (now - last_ctrl_c) < 3:
            # Second Ctrl+C within 3 seconds: actually kill
            signal.signal(signal.SIGINT, original_handler)
            console.print("\n[red]Force quit.[/red]")
            raise KeyboardInterrupt
        else:
            # First Ctrl+C: request pause
            pause_requested = True
            last_ctrl_c = now
            console.print("\n[yellow]Ctrl+C: pausing after current frame... (press again within 3s to force quit)[/yellow]")

    signal.signal(signal.SIGINT, _handle_sigint)
    console.print("[dim]Ctrl+C to pause and save progress[/dim]")

    # Setup output formatter
    out_fmt = OutputFormatter(
        base_dir=str(output), model=name,
        input_name=str(input_path), job_id=job.job_id,
    )
    out_fmt.results.device = resolved_device
    out_fmt.results.variant = job.variant

    # Use the formatter's viz_dir as the actual output for model runners
    run_output_dir = str(out_fmt.viz_dir)

    start = time.time()
    frame = 0
    total = 0
    total_objects = 0
    all_classes = set()
    detections = []  # for COCO export

    try:
        with limiter:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
            ) as progress:
                prog_task = None
                for update in runner_mod.run(loaded_model, str(input_path), run_output_dir,
                                             {"device": resolved_device}):
                    limiter.wait_if_paused()

                    if low_memory:
                        apply_low_memory_cleanup()

                    frame = update.get("frame", 0)
                    total = update.get("total", 0)
                    total_objects = update.get("objects", total_objects)

                    if "classes" in update and isinstance(update["classes"], list):
                        all_classes.update(update["classes"])

                    # Collect detections for COCO export
                    if "detections" in update:
                        detections.extend(update["detections"])

                    # Skip already-processed frames (resume)
                    if frame <= resume_from:
                        if prog_task is None and total > 0:
                            prog_task = progress.add_task(f"Running {name}...", total=total)
                        if prog_task is not None:
                            progress.update(prog_task, completed=frame)
                        continue

                    if prog_task is None and total > 0:
                        prog_task = progress.add_task(f"Running {name}...", total=total)
                    if prog_task is not None:
                        progress.update(prog_task, completed=frame)

                    # Save checkpoint periodically
                    if frame % ckpt_every == 0:
                        job.last_frame = frame
                        job.total_frames = total
                        ckpt_mgr.save_checkpoint(job)

                    # Check if pause was requested via Ctrl+C
                    if pause_requested:
                        job.last_frame = frame
                        job.total_frames = total
                        ckpt_mgr.mark_paused(job)
                        console.print(f"\n[yellow bold]Paused[/yellow bold] at frame {frame}/{total} ({job.progress_percent}%).")
                        console.print(f"Run [cyan]pixo resume[/cyan] to continue.")
                        return

        # Mark completed
        ckpt_mgr.mark_completed(job)

    except KeyboardInterrupt:
        # Hard kill (second Ctrl+C)
        job.last_frame = frame
        job.total_frames = total
        ckpt_mgr.mark_paused(job)
        console.print(f"\n[yellow bold]Paused[/yellow bold] at frame {job.last_frame}/{job.total_frames}.")
        console.print(f"Run [cyan]pixo resume[/cyan] to continue.")
        return

    except Exception as e:
        job.last_frame = frame
        job.total_frames = total
        ckpt_mgr.mark_failed(job)
        console.print(f"\n[red]Failed:[/red] {e}")
        console.print(f"Checkpoint saved. Run [cyan]pixo resume[/cyan] to retry.")
        raise typer.Exit(1)

    finally:
        signal.signal(signal.SIGINT, original_handler)

    elapsed = time.time() - start
    fps = total / elapsed if total and elapsed else 0

    # Save structured output
    out_fmt.set_input_info(str(input_path), frames=total)
    out_fmt.set_timing(elapsed, fps)
    out_fmt.set_summary(
        objects_detected=total_objects,
        classes=sorted(all_classes) if all_classes else [],
    )
    out_fmt.save_results_json()
    out_fmt.save_summary_txt()

    # Export COCO if we have detections
    if detections:
        out_fmt.export_coco(detections, image_width=0, image_height=0)

    # Export CSV summary
    if total_objects:
        csv_rows = [{"frame": f, "objects": total_objects, "classes": ",".join(sorted(all_classes))}
                    for f in range(1, total + 1)] if total <= 100 else []
        if csv_rows:
            out_fmt.export_csv(csv_rows)

    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    console.print()
    console.print("[green bold]Done![/green bold]")
    console.print(f"[bold]Time:[/bold] {time_str}")
    console.print(f"[bold]Device:[/bold] {resolved_device}")
    if total_objects:
        console.print(f"[bold]Objects detected:[/bold] {total_objects}")
    if all_classes:
        console.print(f"[bold]Classes:[/bold] {', '.join(sorted(all_classes))}")
    console.print(f"[bold]Output:[/bold] {out_fmt.root}")


@app.command(name="list")
def list_cmd():
    """List available models."""
    cards = loader.list_cards()
    table = Table(title="Available Models")
    table.add_column("", style="bold")
    table.add_column("Name", style="bold cyan")
    table.add_column("Task", style="green")
    table.add_column("Variants", style="yellow")
    table.add_column("Default Size", justify="right")
    table.add_column("Description")

    for card in cards:
        variant_names = ", ".join(k for k in card.variants.keys() if k != "default")
        default = card.default_variant
        size = f"{default.size_mb} MB" if default.size_mb < 1000 else f"{default.size_mb / 1000:.1f} GB"
        downloaded = "[green]*[/green]" if is_downloaded(card.name, default) else ""
        table.add_row(
            downloaded,
            card.name,
            card.task,
            variant_names or "--",
            size,
            card.description,
        )

    console.print(table)
    console.print(
        "\n[dim]Pull a model:[/dim] pixo pull <name>    "
        "[dim]Variant:[/dim] pixo pull <name>:<variant>"
    )


@app.command()
def info(
    model_name: str = typer.Argument(help="Model name (e.g. sam2, yolov8)"),
):
    """Show detailed info about a model."""
    name, _ = _parse_model_name(model_name)
    card = _get_card(name)

    has_runner = loader.has_runner(name)
    runner_status = "[green]ready[/green]" if has_runner else "[dim]stub[/dim]"

    console.print(Panel(
        f"[bold]{card.name}[/bold] -- {card.description}\n\n"
        f"[bold]Task:[/bold] {card.task}\n"
        f"[bold]Author:[/bold] {card.author}\n"
        f"[bold]Inputs:[/bold] {', '.join(card.input_types)}\n"
        f"[bold]Outputs:[/bold] {', '.join(card.output_types)}\n"
        f"[bold]Source:[/bold] {card.huggingface_repo}\n"
        f"[bold]Runner:[/bold] {runner_status}",
        title=card.name,
    ))

    table = Table(title="Variants")
    table.add_column("Name", style="bold cyan")
    table.add_column("Size", justify="right")
    table.add_column("Min RAM", justify="right")
    table.add_column("Downloaded", justify="center")
    table.add_column("Description")

    for vname, v in card.variants.items():
        size = f"{v.size_mb} MB" if v.size_mb < 1000 else f"{v.size_mb / 1000:.1f} GB"
        dl = "[green]*[/green]" if is_downloaded(card.name, v, vname) else ""
        table.add_row(vname, size, f"{card.hardware.min_ram_gb} GB", dl, v.description)

    console.print(table)
    console.print(
        f"\n[dim]Pull:[/dim] pixo pull {card.name}    "
        f"[dim]Variant:[/dim] pixo pull {card.name}:<variant>"
    )


@app.command()
def doctor():
    """Check hardware and show recommendations."""
    from pixo.core.profiler import get_profile
    from pixo.core.guardian import get_cpu_temperature

    profile = get_profile()

    rows = [
        ("CPU", f"{profile.cpu_name} ({profile.cpu_cores} cores)"),
        ("RAM", f"{profile.ram_total_gb} GB total, {profile.ram_available_gb} GB available"),
    ]

    if profile.has_gpu:
        rows.append(("GPU", f"{profile.gpu_name} ({profile.gpu_vram_gb} GB VRAM)"))
        rows.append(("CUDA", profile.cuda_version or "N/A"))
    else:
        rows.append(("GPU", "Not detected"))

    rows.append(("OS", profile.os_name))
    rows.append(("Disk", f"{profile.disk_free_gb} GB free"))

    temp = get_cpu_temperature()
    if temp is not None:
        if temp < 60:
            temp_status = "idle, normal"
        elif temp < 75:
            temp_status = "normal"
        elif temp < 85:
            temp_status = "warm"
        else:
            temp_status = "HOT"
        rows.append(("CPU Temp", f"{temp:.0f}C ({temp_status})"))
    else:
        rows.append(("CPU Temp", "Not available on this system"))

    info_text = "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in rows)
    console.print(Panel(info_text, title="Hardware Profile"))
    console.print(f"\n[bold]Recommendation:[/bold] {profile.recommendation}")


@app.command()
def optimize(
    model_name: str = typer.Argument(help="Model to optimize (e.g. yolov8, yolov8:small)"),
):
    """Convert a downloaded model to ONNX for faster inference."""
    from pixo.core.optimizer import optimize_model

    name, variant_name = _parse_model_name(model_name)
    card = _get_card(name)
    variant = card.get_variant(variant_name)
    model_path = get_model_path(card.name, variant, variant_name)

    if not model_path.exists():
        console.print(f"[yellow]{model_name} not downloaded. Pulling first...[/yellow]")
        download_model(card, variant_name)

    try:
        optimize_model(name, model_path)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command(name="setup-cloud")
def setup_cloud(
    kaggle: bool = typer.Option(False, "--kaggle", help="Set up Kaggle only"),
    colab: bool = typer.Option(False, "--colab", help="Set up Colab only"),
):
    """Connect your free cloud GPU accounts."""
    from pixo.cloud.config import load_config, save_config

    config = load_config()
    setup_all = not kaggle and not colab

    if setup_all or kaggle:
        console.print(Panel(
            "[bold]Kaggle[/bold] gives you 30hrs/week of free GPU.\n\n"
            "1. Go to https://www.kaggle.com/settings\n"
            "2. Scroll to 'API' section\n"
            "3. Click 'Create New Token' -- downloads kaggle.json\n"
            "4. Open it and copy your username + key",
            title="Kaggle Setup",
        ))
        config.kaggle.username = typer.prompt("Kaggle username")
        config.kaggle.api_key = typer.prompt("Kaggle API key", hide_input=True)
        save_config(config)
        console.print("[green]Kaggle configured![/green]")

    if setup_all or colab:
        console.print(Panel(
            "[bold]Google Colab[/bold] gives you free T4 GPU sessions.\n\n"
            "1. Go to https://colab.research.google.com\n"
            "2. You'll need a Google account\n"
            "3. Paste your Google OAuth token below",
            title="Google Colab Setup",
        ))
        config.colab.token = typer.prompt("Colab token", hide_input=True)
        save_config(config)
        console.print("[green]Colab configured![/green]")

    console.print("\n[bold]Done![/bold] Run [cyan]pixo cloud-status[/cyan] to check connections.")


@app.command(name="cloud-status")
def cloud_status():
    """Show status of connected cloud backends."""
    from pixo.cloud.config import load_config

    config = load_config()

    table = Table(title="Cloud Backends")
    table.add_column("Backend", style="bold cyan")
    table.add_column("Status")
    table.add_column("Details")

    if config.kaggle.is_configured:
        table.add_row("Kaggle", "[green]Connected[/green]", f"User: {config.kaggle.username}")
    else:
        table.add_row("Kaggle", "[dim]Not configured[/dim]", "Run: pixo setup-cloud --kaggle")

    if config.colab.is_configured:
        table.add_row("Colab", "[green]Connected[/green]", "Token saved")
    else:
        table.add_row("Colab", "[dim]Not configured[/dim]", "Run: pixo setup-cloud --colab")

    console.print(table)

    if not config.any_configured:
        console.print("\n[dim]No cloud backends configured. Run:[/dim] pixo setup-cloud")


@app.command()
def history():
    """Show all jobs (running, completed, paused, failed)."""
    from pixo.core.checkpoint import CheckpointManager
    from datetime import datetime

    jobs = CheckpointManager().list_jobs()
    if not jobs:
        console.print("[dim]No jobs yet. Run a model to create one.[/dim]")
        return

    table = Table(title="Job History")
    table.add_column("ID", style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Input")
    table.add_column("Status")
    table.add_column("Progress", justify="right")
    table.add_column("Date")

    status_colors = {
        "completed": "[green]done[/green]",
        "paused": "[yellow]paused[/yellow]",
        "failed": "[red]failed[/red]",
        "running": "[blue]running[/blue]",
    }

    for job in jobs:
        input_name = Path(job.input_path).name if job.input_path else "?"
        status = status_colors.get(job.status, job.status)
        progress = f"{job.progress_percent}%" if job.total_frames else "--"
        date = datetime.fromtimestamp(job.updated_at).strftime("%Y-%m-%d %H:%M") if job.updated_at else "?"
        table.add_row(job.job_id[:8], job.model, input_name, status, progress, date)

    console.print(table)
    console.print("\n[dim]Resume a job:[/dim] pixo resume [job_id]")


@app.command()
def resume(
    job_id: str = typer.Argument(None, help="Job ID to resume (default: most recent)"),
):
    """Resume a paused or failed job."""
    from pixo.core.checkpoint import CheckpointManager
    from pixo.core.runner import get_device

    ckpt_mgr = CheckpointManager()

    if job_id:
        # Find by prefix match
        matches = [j for j in ckpt_mgr.list_jobs() if j.job_id.startswith(job_id)]
        if not matches:
            console.print(f"[red]No job found matching '{job_id}'[/red]")
            raise typer.Exit(1)
        job = matches[0]
    else:
        job = ckpt_mgr.get_latest_resumable()
        if not job:
            console.print("[dim]No paused or failed jobs to resume.[/dim]")
            return

    if job.status not in ("paused", "failed"):
        console.print(f"[yellow]Job {job.job_id[:8]} is {job.status}, not resumable.[/yellow]")
        return

    console.print(f"[bold]Resuming job {job.job_id[:8]}[/bold]: {job.model} on {Path(job.input_path).name}")
    console.print(f"[dim]From frame {job.last_frame}/{job.total_frames} ({job.progress_percent}%)[/dim]")

    # Load card and runner
    card = _get_card(job.model)
    try:
        runner_mod = loader.load_runner(job.model)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    resolved_device = get_device(job.device if job.device != "cpu" else None)
    variant = card.get_variant(None)
    model_dir = str(get_model_path(card.name, variant).parent)

    try:
        loaded_model = runner_mod.setup(model_dir, variant.filename, resolved_device)
    except NotImplementedError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    from pixo.core.guardian import ResourceLimiter
    limiter = ResourceLimiter()
    ckpt_every = card.checkpoint.every if card.checkpoint.supported else 500

    _run_with_checkpoints(
        runner_mod=runner_mod,
        loaded_model=loaded_model,
        name=job.model,
        input_path=Path(job.input_path),
        output=job.output_dir,
        resolved_device=resolved_device,
        limiter=limiter,
        low_memory=False,
        job=job,
        ckpt_mgr=ckpt_mgr,
        ckpt_every=ckpt_every,
        resume_from=job.last_frame,
    )


@app.command(name="jobs-clean")
def jobs_clean():
    """Delete completed job checkpoints to free disk space."""
    from pixo.core.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager()
    count = ckpt_mgr.clean_completed()
    if count:
        console.print(f"[green]Cleaned {count} completed job(s).[/green]")
    else:
        console.print("[dim]No completed jobs to clean.[/dim]")


@app.command()
def view(
    job_id: str = typer.Argument(help="Job ID prefix (from pixo history)"),
):
    """Open job results in your file manager."""
    import subprocess
    import sys
    from pixo.core.checkpoint import CheckpointManager

    mgr = CheckpointManager()
    matches = [j for j in mgr.list_jobs() if j.job_id.startswith(job_id)]
    if not matches:
        console.print(f"[red]No job found matching '{job_id}'[/red]")
        raise typer.Exit(1)

    job = matches[0]
    output_dir = Path(job.output_dir)

    # Find the structured output dir (job_id_model_input)
    if output_dir.exists():
        subdirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith(job.job_id[:8])]
        target = subdirs[0] if subdirs else output_dir
    else:
        console.print(f"[red]Output directory not found: {output_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Opening:[/bold] {target}")

    # Open in system file manager
    if sys.platform == "win32":
        subprocess.run(["explorer", str(target)])
    elif sys.platform == "darwin":
        subprocess.run(["open", str(target)])
    else:
        subprocess.run(["xdg-open", str(target)])


@app.command()
def upgrade():
    """Upgrade pixo to the latest version."""
    import subprocess
    import sys
    from pixo import __version__

    console.print(f"[dim]Current version: {__version__}[/dim]")
    console.print("[bold]Upgrading pixo...[/bold]")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pixo"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        console.print("[green]Upgrade complete![/green]")
    else:
        console.print(f"[red]Upgrade failed:[/red] {result.stderr.strip()}")


@app.command()
def rm(
    model_name: str = typer.Argument(help="Model to remove (e.g. yolov8, sam2:small)"),
):
    """Remove a downloaded model."""
    name, variant_name = _parse_model_name(model_name)
    if remove_model(name, variant_name):
        console.print(f"[green]Removed {model_name}[/green]")
    else:
        console.print(f"[yellow]{model_name} is not downloaded[/yellow]")


@app.command()
def pipe(
    pipeline: str = typer.Argument(help='Pipeline: "grounding_dino -> sam2" or template name'),
    input: str = typer.Option(..., "--input", "-i", help="Input image or video file"),
    output: str = typer.Option("./pixo_output", "--output", "-o", help="Output directory"),
    prompt: str = typer.Option("object", "--prompt", "-p", help="Text prompt for detection models"),
    device: str = typer.Option(None, "--device", "-d", help="Force device: cpu or cuda"),
):
    """Chain multiple models together in a pipeline."""
    from pixo.core.pipeline import parse_pipeline, run_pipeline, list_templates

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]File not found: {input}[/red]")
        raise typer.Exit(1)

    models = parse_pipeline(pipeline)
    if not models:
        console.print("[red]Could not parse pipeline. Use: model1 -> model2 or a template name.[/red]")
        console.print("\n[bold]Available templates:[/bold]")
        for name, steps in list_templates().items():
            console.print(f"  {name}: {' → '.join(steps)}")
        raise typer.Exit(1)

    console.print(f"[bold]Pipeline:[/bold] {' → '.join(models)}")
    console.print(f"[bold]Input:[/bold] {input_path}")

    options = {"prompt": prompt}
    run_pipeline(models, input_path, Path(output), options, device=device or "cpu")

    console.print(f"\n[bold green]Pipeline complete![/bold green] Results in: {output}")


@app.command(name="env-list")
def env_list():
    """Show all isolated model environments."""
    from pixo.core.envmanager import list_envs

    envs = list_envs()
    if not envs:
        console.print("[dim]No isolated environments. Use: pixo pull <model> --isolate[/dim]")
        return

    table = Table(title="Model Environments")
    table.add_column("Model", style="bold cyan")
    table.add_column("Size", justify="right")
    table.add_column("Path")

    for env in envs:
        size = f"{env['size_mb']} MB" if env['size_mb'] < 1000 else f"{env['size_mb'] / 1000:.1f} GB"
        table.add_row(env["name"], size, env["path"])

    console.print(table)


@app.command(name="env-clean")
def env_clean(
    model_name: str = typer.Argument(help="Model environment to remove"),
):
    """Remove and rebuild a model's isolated environment."""
    from pixo.core.envmanager import delete_env

    if delete_env(model_name):
        console.print(f"[green]Removed environment for {model_name}[/green]")
        console.print(f"[dim]Recreate with: pixo pull {model_name} --isolate[/dim]")
    else:
        console.print(f"[yellow]No environment found for {model_name}[/yellow]")
