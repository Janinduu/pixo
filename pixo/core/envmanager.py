"""Environment manager — per-model Python venv isolation.

Creates isolated virtual environments for each model so their
dependencies never conflict with each other.

Usage modes:
- Default: models run in the user's current Python environment
- --isolate flag: creates a venv at ~/.pixo/envs/<model>/ and runs there
"""

import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

ENVS_DIR = Path.home() / ".pixo" / "envs"


def _get_env_dir(model_name: str) -> Path:
    """Return the venv directory for a model."""
    return ENVS_DIR / model_name


def _get_python(model_name: str) -> Path:
    """Return the Python binary inside a model's venv."""
    env_dir = _get_env_dir(model_name)
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def env_exists(model_name: str) -> bool:
    """Check if an isolated environment exists for a model."""
    return _get_python(model_name).exists()


def create_env(model_name: str) -> Path:
    """Create a virtual environment for a model. Returns the env directory."""
    env_dir = _get_env_dir(model_name)

    if env_exists(model_name):
        console.print(f"[dim]Environment for {model_name} already exists.[/dim]")
        return env_dir

    env_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Creating environment for {model_name}...[/bold]")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(env_dir)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        console.print(f"[red]Failed to create venv: {result.stderr}[/red]")
        raise RuntimeError(f"venv creation failed for {model_name}")

    # Upgrade pip in the new env
    python = str(_get_python(model_name))
    subprocess.run(
        [python, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True, text=True,
    )

    return env_dir


def install_deps(model_name: str, packages: list[str]) -> bool:
    """Install dependencies into a model's venv. Returns True on success."""
    python = str(_get_python(model_name))

    if not packages:
        return True

    console.print(f"[bold]Installing dependencies for {model_name}...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Installing {len(packages)} packages...", total=None)

        result = subprocess.run(
            [python, "-m", "pip", "install"] + packages,
            capture_output=True, text=True,
            timeout=600,  # 10 minute timeout
        )

        progress.update(task, completed=True)

    if result.returncode != 0:
        console.print(f"[red]Failed to install dependencies:[/red]")
        # Show the last few lines of error output
        error_lines = result.stderr.strip().split("\n")
        for line in error_lines[-5:]:
            console.print(f"  [dim]{line}[/dim]")
        return False

    console.print(f"[green]Dependencies installed.[/green]")
    return True


def run_in_env(model_name: str, script: str, args: dict) -> subprocess.Popen:
    """Run a Python script string inside a model's venv.

    Returns a Popen object. Caller should read stdout for progress JSON lines.
    Each line of stdout should be a JSON dict with progress info.
    """
    python = str(_get_python(model_name))

    # Pass args as JSON via stdin
    args_json = json.dumps(args)

    proc = subprocess.Popen(
        [python, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send args via stdin, then close it
    if proc.stdin:
        proc.stdin.write(args_json)
        proc.stdin.close()

    return proc


def delete_env(model_name: str) -> bool:
    """Delete a model's environment. Returns True if something was deleted."""
    import shutil
    env_dir = _get_env_dir(model_name)
    if env_dir.exists():
        shutil.rmtree(env_dir)
        return True
    return False


def list_envs() -> list[dict]:
    """List all model environments with their sizes."""
    if not ENVS_DIR.exists():
        return []

    envs = []
    for d in sorted(ENVS_DIR.iterdir()):
        if d.is_dir() and (d / "Scripts" / "python.exe").exists() or (d / "bin" / "python").exists():
            # Calculate size
            total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            size_mb = round(total_size / (1024 * 1024), 1)
            envs.append({
                "name": d.name,
                "path": str(d),
                "size_mb": size_mb,
            })
    return envs


def get_env_wrapper_script(run_py_path: str) -> str:
    """Generate a wrapper script that imports a model's run.py and executes it.

    The wrapper reads args from stdin (JSON), calls setup() then run(),
    and writes progress dicts to stdout as JSON lines.
    """
    return f'''
import sys
import json

# Read args from stdin
args = json.loads(sys.stdin.read() if not sys.stdin.isatty() else "{{}}")

# Import the runner
import importlib.util
spec = importlib.util.spec_from_file_location("runner", r"{run_py_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Setup
model = mod.setup(
    args.get("model_dir", ""),
    args.get("variant", ""),
    args.get("device", "cpu"),
)

# Run and stream progress as JSON lines
for progress in mod.run(model, args["input_path"], args["output_dir"], args.get("options", {{}})):
    print(json.dumps(progress), flush=True)
'''
