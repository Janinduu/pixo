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
    help="Run any computer vision model with one command.",
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool):
    if value:
        from pixo import __version__
        console.print(f"pixo {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    """pixo — run any computer vision model with one command."""


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
    prompt: str = typer.Option(None, "--prompt", "-p", help="Text prompt for detection models (e.g. 'person, car')"),
    task: str = typer.Option(None, "--task", help="Task for multi-task models like Florence-2 (caption, detect, ocr)"),
    airgap: bool = typer.Option(False, "--airgap", help="Block all outbound network calls during the run"),
):
    """Run inference on an image or video."""
    from pixo.core.optimizer import get_optimized_path, is_optimized
    from pixo.core.guardian import check_can_run, display_safety_check
    from pixo.core.runner import get_device
    from pixo.cloud.config import load_config as load_cloud_config
    from pixo.cloud.router import pick_backend
    from pixo.core.airgap import airgap_enforced, AirgapViolation

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]File not found: {input}[/red]")
        raise typer.Exit(1)

    # Airgap is incompatible with cloud backends
    if airgap and backend in ("kaggle", "colab"):
        console.print("[red]--airgap cannot be combined with cloud backends[/red]")
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

    # Model-specific hints
    if name == "grounding_dino" and not prompt:
        console.print("\n[yellow]Hint:[/yellow] GroundingDINO needs a text prompt to know what to detect.")
        console.print("  Add: [cyan]--prompt \"person, car, dog\"[/cyan]")
        console.print("  Without it, detection results will be poor.\n")
    if name == "florence2" and not task:
        console.print("\n[yellow]Hint:[/yellow] Florence-2 supports multiple tasks.")
        console.print("  [cyan]--task caption[/cyan]   Describe the image")
        console.print("  [cyan]--task detect[/cyan]    Find objects")
        console.print("  [cyan]--task ocr[/cyan]       Extract text\n")
    if name in ("sam2", "samurai") and resolved_device == "cpu":
        console.print("\n[yellow]Hint:[/yellow] SAM2/SAMURAI is very slow on CPU (10-45 min per image).")
        console.print("  For faster results: [cyan]--backend kaggle[/cyan] (free GPU, ~1 min)")
        console.print("  Or use a smaller variant: [cyan]pixo pull sam2:tiny[/cyan]\n")

    console.print(f"[bold]Device:[/bold] {resolved_device}")
    console.print(f"[bold]Input:[/bold] {input_path}")
    if airgap:
        console.print("[bold magenta]Airgap:[/bold magenta] network access blocked for this run")

    def _do_run():
        loaded_model = runner_mod.setup(model_dir, variant.filename, resolved_device)

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
            prompt=prompt,
            task=task,
        )

    try:
        if airgap:
            with airgap_enforced():
                _do_run()
        else:
            _do_run()
    except AirgapViolation as e:
        console.print(f"\n[red]{e}[/red]")
        raise typer.Exit(1)
    except (NotImplementedError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


def _run_with_checkpoints(runner_mod, loaded_model, name, input_path, output,
                          resolved_device, limiter, low_memory, job, ckpt_mgr,
                          ckpt_every, resume_from, prompt=None, task=None):
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
                run_options = {"device": resolved_device}
                if prompt:
                    run_options["prompt"] = prompt
                if task:
                    run_options["task"] = task
                for update in runner_mod.run(loaded_model, str(input_path), run_output_dir,
                                             run_options):
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


_PRIVACY_COLORS = {"green": "green", "yellow": "yellow", "red": "red"}


def _privacy_badge(level: str) -> str:
    color = _PRIVACY_COLORS.get(level, "dim")
    return f"[{color}]{level}[/{color}]"


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
    table.add_column("Privacy", justify="center")
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
            _privacy_badge(card.privacy.level),
            card.description,
        )

    console.print(table)
    console.print(
        "\n[dim]Privacy:[/dim] "
        "[green]green[/green] = offline after pull   "
        "[yellow]yellow[/yellow] = needs first-pull internet   "
        "[red]red[/red] = needs runtime internet"
    )
    console.print(
        "[dim]Pull a model:[/dim] pixo pull <name>    "
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

    privacy_line = _privacy_badge(card.privacy.level)
    if card.privacy.note:
        privacy_line += f" [dim]— {card.privacy.note}[/dim]"

    console.print(Panel(
        f"[bold]{card.name}[/bold] -- {card.description}\n\n"
        f"[bold]Task:[/bold] {card.task}\n"
        f"[bold]Author:[/bold] {card.author}\n"
        f"[bold]Inputs:[/bold] {', '.join(card.input_types)}\n"
        f"[bold]Outputs:[/bold] {', '.join(card.output_types)}\n"
        f"[bold]Source:[/bold] {card.huggingface_repo}\n"
        f"[bold]Privacy:[/bold] {privacy_line}\n"
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
    except (NotImplementedError, RuntimeError) as e:
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
def guide():
    """Show a complete guide on how to use pixo."""
    from pixo import __version__

    console.print(Panel(
        f"[bold]pixo v{__version__}[/bold]\n"
        "Run any computer vision model with one command -- without freezing your laptop.",
        title="Welcome to pixo",
        border_style="cyan",
    ))

    # Getting Started
    console.print("\n[bold cyan]Getting Started[/bold cyan]\n")
    console.print("  [bold]1.[/bold] See available models:     [cyan]pixo list[/cyan]")
    console.print("  [bold]2.[/bold] Check your hardware:      [cyan]pixo doctor[/cyan]")
    console.print("  [bold]3.[/bold] Download a model:         [cyan]pixo pull yolov8[/cyan]")
    console.print("  [bold]4.[/bold] Run on an image:          [cyan]pixo run yolov8 --input photo.jpg[/cyan]")
    console.print("  [bold]5.[/bold] Run on a video:           [cyan]pixo run yolov8 --input video.mp4[/cyan]")

    # Models
    console.print("\n[bold cyan]Available Models[/bold cyan]\n")
    model_table = Table(show_header=True, header_style="bold")
    model_table.add_column("Model", style="cyan")
    model_table.add_column("What it does")
    model_table.add_column("Speed (CPU)")
    model_table.add_column("Example")
    model_table.add_row("yolov8", "Detect objects (person, car...)", "Fast (~7s)", "pixo run yolov8 -i photo.jpg")
    model_table.add_row("yolov11", "Detection (newer, more accurate)", "Fast (~6s)", "pixo run yolov11 -i photo.jpg")
    model_table.add_row("yolov12", "Detection (latest, attention-based)", "Fast (~6s)", "pixo run yolov12 -i photo.jpg")
    model_table.add_row("rtdetr", "Detection (transformer, no NMS)", "Fast (~10s)", "pixo run rtdetr -i photo.jpg")
    model_table.add_row("grounding_dino", "Detect anything by text", "Medium (~45s)", "pixo run grounding_dino -i photo.jpg -p \"person\"")
    model_table.add_row("depth_anything_v2", "Estimate depth (near/far)", "Fast (~3s)", "pixo run depth_anything_v2 -i photo.jpg")
    model_table.add_row("sam2", "Segment everything", "Slow (use cloud)", "pixo run sam2 -i photo.jpg")
    model_table.add_row("samurai", "Track + segment in video", "Slow (use cloud)", "pixo run samurai -i video.mp4")
    model_table.add_row("florence2", "Caption, detect, OCR", "Needs transformers<5", "pixo run florence2 -i photo.jpg --task caption")
    console.print(model_table)

    # Performance
    console.print("\n[bold cyan]Performance Tips[/bold cyan]\n")
    console.print("  [bold]Laptop freezing?[/bold]")
    console.print("    [cyan]pixo run yolov8 -i video.mp4 --low-memory[/cyan]    Process frame-by-frame, less RAM")
    console.print("    [cyan]pixo run yolov8 -i video.mp4 --background[/cyan]    Lowest priority, laptop stays usable")
    console.print()
    console.print("  [bold]Too slow on CPU?[/bold]")
    console.print("    [cyan]pixo optimize yolov8[/cyan]                         Convert to ONNX (40% faster)")
    console.print("    [cyan]pixo setup-cloud --kaggle[/cyan]                    Connect free Kaggle GPU (30hrs/week)")
    console.print("    [cyan]pixo run sam2 -i photo.jpg --backend kaggle[/cyan]  Run on cloud GPU instead")
    console.print()
    console.print("  [bold]Model variants (smaller = faster):[/bold]")
    console.print("    [cyan]pixo pull sam2:tiny[/cyan]                          Smallest, fastest")
    console.print("    [cyan]pixo pull sam2:small[/cyan]                         Good balance")
    console.print("    [cyan]pixo pull sam2[/cyan]                               Full quality (default)")

    # Cloud GPUs
    console.print("\n[bold cyan]Free Cloud GPUs[/bold cyan]\n")
    console.print("  Kaggle gives you [bold]30 hours/week[/bold] of free T4 GPU.")
    console.print("  SAM2 on Kaggle GPU: ~1 min. SAM2 on CPU: ~44 min.\n")
    console.print("  [bold]Setup (one time):[/bold]")
    console.print("    1. Go to https://www.kaggle.com/settings")
    console.print("    2. Scroll to API > Create New Token")
    console.print("    3. Run: [cyan]pixo setup-cloud --kaggle[/cyan]")
    console.print("    4. Now use: [cyan]pixo run <model> --backend kaggle[/cyan]")

    # Checkpointing
    console.print("\n[bold cyan]Never Lose Progress[/bold cyan]\n")
    console.print("  pixo auto-saves progress every 100 frames.\n")
    console.print("  [bold]Ctrl+C[/bold]        Pause (saves checkpoint, doesn't kill)")
    console.print("  [bold]Ctrl+C x2[/bold]     Actually quit")
    console.print("  [cyan]pixo resume[/cyan]    Pick up from where you stopped")
    console.print("  [cyan]pixo history[/cyan]   See all past jobs and their status")

    # Model Piping
    console.print("\n[bold cyan]Chain Models Together[/bold cyan]\n")
    console.print("  Run multiple models in sequence on the same input:\n")
    console.print("    [cyan]pixo pipe \"yolov8 -> depth_anything_v2\" -i photo.jpg[/cyan]")
    console.print("    [cyan]pixo pipe \"grounding_dino -> sam2\" -i photo.jpg --prompt \"person\"[/cyan]")
    console.print()
    console.print("  [bold]Pre-built templates:[/bold]")
    console.print("    [cyan]pixo pipe detect_and_segment -i photo.jpg --prompt \"car\"[/cyan]")
    console.print("    [cyan]pixo pipe segment_and_depth -i photo.jpg[/cyan]")

    # Model-specific
    console.print("\n[bold cyan]Model-Specific Options[/bold cyan]\n")
    console.print("  [bold]grounding_dino[/bold] -- needs a text prompt:")
    console.print("    [cyan]--prompt \"person, car, dog\"[/cyan]   What to detect (required)")
    console.print()
    console.print("  [bold]florence2[/bold] -- multi-task model:")
    console.print("    [cyan]--task caption[/cyan]               Describe the image")
    console.print("    [cyan]--task detailed_caption[/cyan]      Detailed description")
    console.print("    [cyan]--task detect[/cyan]                Find objects")
    console.print("    [cyan]--task ocr[/cyan]                   Extract text")

    # Output
    console.print("\n[bold cyan]Understanding Output[/bold cyan]\n")
    console.print("  Every run creates a structured output folder:\n")
    console.print("    pixo_output/")
    console.print("      results.json          Machine-readable metadata")
    console.print("      summary.txt           Human-readable summary")
    console.print("      visualizations/       Annotated images or video")
    console.print("      exports/              COCO JSON + CSV exports")
    console.print()
    console.print("  [cyan]pixo view <job_id>[/cyan]   Open the output folder")

    # All Commands
    console.print("\n[bold cyan]All Commands[/bold cyan]\n")
    cmd_table = Table(show_header=True, header_style="bold")
    cmd_table.add_column("Command", style="cyan")
    cmd_table.add_column("What it does")
    cmd_table.add_row("pixo list", "Show all available models")
    cmd_table.add_row("pixo info <model>", "Detailed model info and variants")
    cmd_table.add_row("pixo pull <model>", "Download a model")
    cmd_table.add_row("pixo run <model> -i <file>", "Run inference")
    cmd_table.add_row("pixo pipe \"m1 -> m2\" -i <file>", "Chain models together")
    cmd_table.add_row("pixo doctor", "Check your hardware")
    cmd_table.add_row("pixo optimize <model>", "Convert to ONNX (faster)")
    cmd_table.add_row("pixo history", "Show all past jobs")
    cmd_table.add_row("pixo resume [job_id]", "Resume a paused/failed job")
    cmd_table.add_row("pixo view <job_id>", "Open job results")
    cmd_table.add_row("pixo setup-cloud", "Connect Kaggle/Colab")
    cmd_table.add_row("pixo cloud-status", "Check cloud connections")
    cmd_table.add_row("pixo rm <model>", "Remove a downloaded model")
    cmd_table.add_row("pixo upgrade", "Update pixo to latest version")
    cmd_table.add_row("pixo guide", "Show this guide")
    console.print(cmd_table)

    console.print("\n[dim]More info: https://github.com/Janinduu/pixo[/dim]\n")


@app.command()
def ui(
    port: int = typer.Option(8420, "--port", "-p", help="Port to run the server on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
):
    """Start the web dashboard (requires: pip install pixo[web])."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Web dashboard requires extra dependencies.[/red]")
        console.print("Install with: [cyan]pip install pixo[web][/cyan]")
        raise typer.Exit(1)

    console.print(f"[bold]Starting pixo dashboard...[/bold]")
    console.print(f"[cyan]http://localhost:{port}[/cyan]")
    console.print(f"[cyan]API docs: http://localhost:{port}/docs[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    if not no_browser:
        import webbrowser
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    uvicorn.run("pixo.server.app:app", host="0.0.0.0", port=port, reload=False)


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
            console.print(f"  {name}: {' -> '.join(steps)}")
        raise typer.Exit(1)

    console.print(f"[bold]Pipeline:[/bold] {' -> '.join(models)}")
    console.print(f"[bold]Input:[/bold] {input_path}")

    options = {"prompt": prompt}
    run_pipeline(models, input_path, Path(output), options, device=device or "cpu")

    console.print(f"\n[bold green]Pipeline complete![/bold green] Results in: {output}")


@app.command()
def serve(
    model_name: str = typer.Argument(help="Model to serve (e.g. yolov8, grounding_dino)"),
    port: int = typer.Option(7860, "--port", "-p", help="Port to run the UI on"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link (requires internet)"),
):
    """Launch a browser UI for one model — drag-drop an image and see results instantly."""
    try:
        import gradio as gr  # type: ignore
    except ImportError:
        console.print("[red]Gradio not installed.[/red]")
        console.print("Install with: [cyan]pip install pixo[demo][/cyan]")
        raise typer.Exit(1)

    name, variant_name = _parse_model_name(model_name)
    card = _get_card(name)
    variant = card.get_variant(variant_name)
    model_path = get_model_path(card.name, variant, variant_name)

    if not model_path.exists():
        console.print(f"[yellow]Pulling {model_name}...[/yellow]")
        download_model(card, variant_name)

    from pixo.core.runner import get_device
    resolved_device = get_device(None)

    try:
        runner_mod = loader.load_runner(name)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Loading {model_name} on {resolved_device}...[/dim]")
    loaded_model = runner_mod.setup(str(model_path.parent), variant.filename, resolved_device)

    import tempfile

    def _infer(image, prompt_text: str | None, task_choice: str | None):
        """Gradio callback. Runs the model on the uploaded image."""
        if image is None:
            return None, "Please upload an image."

        # Save the uploaded image to a temp file (gradio gives us a path or PIL)
        if isinstance(image, str):
            input_path = Path(image)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            image.save(tmp.name)
            input_path = Path(tmp.name)

        out_dir = Path(tempfile.mkdtemp(prefix="pixo_serve_"))
        options = {"device": resolved_device}
        if prompt_text:
            options["prompt"] = prompt_text
        if task_choice and task_choice != "(none)":
            options["task"] = task_choice

        summary: dict = {}
        for update in runner_mod.run(loaded_model, str(input_path), str(out_dir), options):
            summary = update

        # Find the visualization
        images = sorted([p for p in out_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
        viz = str(images[-1]) if images else None

        summary_text = "\n".join(f"{k}: {v}" for k, v in summary.items() if k not in ("frame", "total"))
        return viz, summary_text or "Done."

    needs_prompt = name in ("grounding_dino",)
    needs_task = name in ("florence2",)

    with gr.Blocks(title=f"pixo — {model_name}") as demo:
        gr.Markdown(f"## pixo — `{model_name}`\n{card.description}")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input image")
                prompt_in = gr.Textbox(
                    label="Prompt (required)" if needs_prompt else "Prompt (optional)",
                    placeholder="person, car, dog" if needs_prompt else "",
                    visible=needs_prompt,
                )
                task_in = gr.Dropdown(
                    choices=["(none)", "caption", "detailed_caption", "detect", "ocr"],
                    value="(none)",
                    label="Task",
                    visible=needs_task,
                )
                btn = gr.Button("Run", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Result")
                out_text = gr.Textbox(label="Summary", lines=6)
        btn.click(_infer, inputs=[inp, prompt_in, task_in], outputs=[out_img, out_text])

    console.print(f"[bold]UI:[/bold] http://localhost:{port}")
    demo.launch(server_port=port, share=share, inbrowser=True)


@app.command()
def compare(
    models: list[str] = typer.Argument(help="Two or more detection models (e.g. yolov8 yolov11 yolov12)"),
    input: str = typer.Option(..., "--input", "-i", help="Input image"),
    output: str = typer.Option("./pixo_output", "--output", "-o", help="Where to save the report"),
    conf: float = typer.Option(0.25, "--conf", help="Confidence threshold"),
    iou_threshold: float = typer.Option(0.5, "--iou", help="IoU threshold for matching boxes across models"),
    device: str = typer.Option(None, "--device", "-d", help="Force device: cpu or cuda"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the report in your browser"),
):
    """Run multiple detection models on one image and see only where they disagree."""
    from pixo.core.compare import (
        detect_with_model, group_detections, classify_groups,
        build_compare_html, SUPPORTED_MODELS,
    )
    from pixo.core.runner import get_device

    if len(models) < 2:
        console.print("[red]Need at least two models to compare. Example: pixo compare yolov8 yolov11 --input photo.jpg[/red]")
        raise typer.Exit(1)

    input_path = Path(input)
    if not input_path.exists():
        console.print(f"[red]File not found: {input}[/red]")
        raise typer.Exit(1)

    unsupported = [m for m in models if m not in SUPPORTED_MODELS]
    if unsupported:
        console.print(f"[red]Unsupported model(s): {', '.join(unsupported)}[/red]")
        console.print(f"[dim]v0.3 compare supports: {', '.join(sorted(SUPPORTED_MODELS))}[/dim]")
        raise typer.Exit(1)

    resolved_device = get_device(device)
    console.print(f"[bold]Comparing:[/bold] {' vs '.join(models)}")
    console.print(f"[bold]Input:[/bold] {input_path}")
    console.print(f"[bold]Device:[/bold] {resolved_device}")
    console.print()

    detections_by_model: dict = {}
    image_size: tuple[int, int] | None = None
    for m in models:
        console.print(f"[dim]Running {m}...[/dim]")
        try:
            dets, size = detect_with_model(m, input_path, conf=conf, device=resolved_device)
        except Exception as e:
            console.print(f"[red]Failed running {m}: {e}[/red]")
            raise typer.Exit(1)
        image_size = image_size or size
        detections_by_model[m] = dets
        console.print(f"  {m}: {len(dets)} detections")

    groups = group_detections(detections_by_model, iou_threshold=iou_threshold)
    classified = classify_groups(groups, models)

    console.print()
    console.print(f"[green]Agreements (all models):[/green] {len(classified['agreements'])}")
    console.print(f"[yellow]Partial agreement:[/yellow] {len(classified['partials'])}")
    console.print(f"[red]Only one model:[/red] {len(classified['uniques'])}")

    # Write report
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_name = f"compare_{'_vs_'.join(models)}_{input_path.stem}.html"
    report_path = output_root / report_name
    report_path.write_text(
        build_compare_html(input_path, models, groups, image_size),
        encoding="utf-8",
    )
    console.print(f"\n[bold]Report:[/bold] {report_path}")
    console.print("[dim]Self-contained HTML. Attach to a tweet or Slack — no server needed.[/dim]")

    if open_browser:
        import webbrowser
        webbrowser.open(report_path.as_uri())


@app.command()
def share(
    job_id: str = typer.Argument(None, help="Job ID prefix to share (default: most recent)"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the report in your browser"),
):
    """Create a self-contained HTML report for a run. Opens in any browser, no network needed."""
    from pixo.core.share import create_share_bundle, find_run_dir_by_job
    from pixo.core.checkpoint import CheckpointManager

    mgr = CheckpointManager()
    if job_id:
        matches = [j for j in mgr.list_jobs() if j.job_id.startswith(job_id)]
        if not matches:
            console.print(f"[red]No job found matching '{job_id}'[/red]")
            raise typer.Exit(1)
        target_job = matches[0]
    else:
        jobs = [j for j in mgr.list_jobs() if j.status == "completed"]
        if not jobs:
            console.print("[dim]No completed jobs to share. Run a model first.[/dim]")
            raise typer.Exit(1)
        target_job = jobs[0]  # most recent

    run_dir = find_run_dir_by_job(target_job.job_id)
    if not run_dir:
        console.print(f"[red]Output directory not found for job {target_job.job_id[:8]}[/red]")
        raise typer.Exit(1)

    try:
        html_path = create_share_bundle(run_dir)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    size_mb = html_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]Share bundle created:[/green] {html_path} ([dim]{size_mb:.1f} MB, self-contained[/dim])")
    console.print("[dim]Attach this file to a tweet, email, Slack, or GitHub issue. No server needed.[/dim]")

    if open_browser:
        import webbrowser
        webbrowser.open(html_path.as_uri())


@app.command(name="try")
def try_cmd(
    model: str = typer.Option(None, "--model", "-m", help="Model to try (default: auto-pick based on hardware)"),
    input: str = typer.Option(None, "--input", "-i", help="Input image (default: bundled sample)"),
):
    """Run a zero-setup demo to see pixo in action — picks a model, finds a sample image, opens a report."""
    from pixo.core.sample import get_sample_image
    from pixo.core.profiler import get_profile
    from pixo.core.share import create_share_bundle, find_run_dir_by_job
    from pixo.core.checkpoint import CheckpointManager

    console.print(Panel(
        "[bold]pixo try[/bold] — one command, zero setup.\n"
        "Picking a model and a sample image, then running it on your machine.",
        border_style="cyan",
    ))

    # 1. Pick a sample image
    if input:
        input_path = Path(input)
        if not input_path.exists():
            console.print(f"[red]File not found: {input}[/red]")
            raise typer.Exit(1)
    else:
        sample = get_sample_image()
        if not sample:
            console.print(
                "[red]Could not find or download a sample image.[/red]\n"
                "Provide one with: [cyan]pixo try --input your_photo.jpg[/cyan]"
            )
            raise typer.Exit(1)
        input_path = sample
        console.print(f"[dim]Sample:[/dim] {input_path.name}")

    # 2. Pick a model based on hardware
    if not model:
        profile = get_profile()
        if profile.has_gpu and profile.gpu_vram_gb and profile.gpu_vram_gb >= 4:
            model = "yolov11"
        elif profile.ram_total_gb >= 8:
            model = "yolov8"
        else:
            model = "yolov8"  # smallest variant works on 2GB
        console.print(f"[dim]Model:[/dim] {model} (auto-picked for your hardware)")

    # 3. Auto-pull if needed and run
    name, variant_name = _parse_model_name(model)
    card = _get_card(name)
    variant = card.get_variant(variant_name)
    model_path = get_model_path(card.name, variant, variant_name)
    if not model_path.exists():
        console.print(f"[yellow]Pulling {model}...[/yellow]")
        download_model(card, variant_name)

    console.print()

    # Call the run command function directly with minimal args
    output_dir = str(Path("./pixo_output").resolve())

    try:
        run(
            model_name=model,
            input=str(input_path),
            output=output_dir,
            device=None,
            backend="local",
            force=False,
            max_ram=None,
            max_cpu=None,
            low_memory=False,
            background=False,
            isolate=False,
            prompt=None,
            task=None,
            airgap=False,
        )
    except typer.Exit:
        pass

    # 4. Find the most recent completed job and open a share bundle
    mgr = CheckpointManager()
    completed = [j for j in mgr.list_jobs() if j.status == "completed" and j.model == name]
    if not completed:
        console.print("\n[yellow]Run finished but no completed job was recorded.[/yellow]")
        return

    latest = completed[0]
    run_dir = find_run_dir_by_job(latest.job_id)
    if not run_dir:
        return

    try:
        html_path = create_share_bundle(run_dir)
    except FileNotFoundError:
        return

    console.print()
    console.print(Panel(
        f"[bold green]Done![/bold green] Your first pixo run is ready.\n\n"
        f"Report: {html_path}\n"
        f"Share it: attach the HTML file to a tweet or Slack — it's self-contained.\n\n"
        f"Try more:\n"
        f"  [cyan]pixo list[/cyan]                     See all models\n"
        f"  [cyan]pixo run yolov8 -i <file>[/cyan]    Run on your own file\n"
        f"  [cyan]pixo compare yolov8 yolov11 -i <file>[/cyan]  Compare two models",
        title="Welcome to pixo",
        border_style="green",
    ))

    import webbrowser
    webbrowser.open(html_path.as_uri())


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
