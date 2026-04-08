"""Resource Guardian — prevents pixo from freezing your laptop.

Three jobs:
1. Pre-run check: can this model run safely on this hardware?
2. Runtime limits: cap RAM/CPU/GPU during execution
3. Suggestions: offer alternatives when resources are tight
"""

import gc
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import psutil
from rich.console import Console
from rich.panel import Panel

console = Console()


# --- System state ---

@dataclass
class SystemState:
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    cpu_count: int
    cpu_percent: float
    gpu_name: str | None = None
    gpu_vram_total_gb: float | None = None
    gpu_vram_free_gb: float | None = None
    disk_free_gb: float = 0.0


def get_system_state() -> SystemState:
    """Snapshot of current system resources."""
    mem = psutil.virtual_memory()

    gpu_name = None
    gpu_vram_total = None
    gpu_vram_free = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_total = round(props.total_mem / (1024 ** 3), 1)
            free, _ = torch.cuda.mem_get_info(0)
            gpu_vram_free = round(free / (1024 ** 3), 1)
    except (ImportError, Exception):
        pass

    pixo_dir = Path.home() / ".pixo"
    try:
        import shutil
        disk_free = shutil.disk_usage(pixo_dir if pixo_dir.exists() else Path.home()).free
        disk_free_gb = round(disk_free / (1024 ** 3), 1)
    except Exception:
        disk_free_gb = 0.0

    return SystemState(
        ram_total_gb=round(mem.total / (1024 ** 3), 1),
        ram_available_gb=round(mem.available / (1024 ** 3), 1),
        ram_used_percent=mem.percent,
        cpu_count=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        cpu_percent=psutil.cpu_percent(interval=0.5),
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total,
        gpu_vram_free_gb=gpu_vram_free,
        disk_free_gb=disk_free_gb,
    )


# --- Model needs estimation ---

@dataclass
class ResourceEstimate:
    ram_needed_gb: float
    gpu_vram_needed_gb: float
    description: str


def estimate_model_needs(model_name: str, input_path: Path, model_size_mb: int) -> ResourceEstimate:
    """Estimate how much RAM/GPU the model will need for this input."""
    suffix = input_path.suffix.lower()
    is_video = suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")
    input_size_mb = input_path.stat().st_size / (1024 * 1024)

    # Base: model weights in RAM
    model_ram_gb = model_size_mb / 1024

    if is_video:
        # Video needs: model + frame buffers + processing overhead
        # Rough: model size * 3 for buffers, + input decoding overhead
        ram_needed = model_ram_gb * 3 + min(input_size_mb / 1024, 2.0)
        desc = f"Video processing: model ({model_ram_gb:.1f}GB) + frame buffers + decode overhead"
    else:
        # Image: model + single frame + output
        ram_needed = model_ram_gb * 2 + 0.5
        desc = f"Image processing: model ({model_ram_gb:.1f}GB) + frame + output buffers"

    # GPU VRAM: roughly model size + activations
    gpu_needed = model_ram_gb * 1.5 if model_ram_gb > 0.1 else 0

    return ResourceEstimate(
        ram_needed_gb=round(ram_needed, 1),
        gpu_vram_needed_gb=round(gpu_needed, 1),
        description=desc,
    )


# --- Safety check ---

@dataclass
class SafetyResult:
    safe: bool
    level: str  # "safe", "borderline", "dangerous"
    message: str
    ram_ok: bool
    gpu_ok: bool
    suggestions: list[str]


def check_can_run(
    model_name: str,
    input_path: Path,
    model_size_mb: int,
    state: SystemState | None = None,
) -> SafetyResult:
    """Pre-run safety check. Returns whether it's safe to proceed."""
    if state is None:
        state = get_system_state()

    needs = estimate_model_needs(model_name, input_path, model_size_mb)

    # RAM check
    ram_headroom = state.ram_available_gb - needs.ram_needed_gb
    ram_ok = ram_headroom > 0.5  # need at least 0.5GB headroom for OS

    # GPU check (always OK if no GPU needed or no GPU present)
    gpu_ok = True
    if state.gpu_name and needs.gpu_vram_needed_gb > 0:
        if state.gpu_vram_free_gb and state.gpu_vram_free_gb < needs.gpu_vram_needed_gb:
            gpu_ok = False

    suggestions = []

    if ram_ok and gpu_ok:
        return SafetyResult(
            safe=True,
            level="safe",
            message=f"Resources OK: need ~{needs.ram_needed_gb}GB RAM, {state.ram_available_gb}GB available.",
            ram_ok=True,
            gpu_ok=True,
            suggestions=[],
        )

    # Build suggestions
    if not ram_ok:
        deficit = needs.ram_needed_gb - state.ram_available_gb
        suggestions.append(f"Close some apps to free ~{deficit:.1f}GB RAM and retry")
        suggestions.append("Use --low-memory mode (slower but safe)")
        suggestions.append("Route to cloud GPU instead (--backend kaggle)")

    if not gpu_ok:
        suggestions.append("Use --device cpu to run on CPU instead")
        suggestions.append("Close GPU-heavy apps (games, browsers with hardware accel)")

    # Borderline: tight but might work
    if ram_headroom > -1.0 and ram_headroom <= 0.5:
        return SafetyResult(
            safe=False,
            level="borderline",
            message=(
                f"Tight on resources: model needs ~{needs.ram_needed_gb}GB RAM, "
                f"you have {state.ram_available_gb}GB free.\n"
                f"This might work but could slow down your machine."
            ),
            ram_ok=False,
            gpu_ok=gpu_ok,
            suggestions=suggestions,
        )

    # Dangerous: will definitely freeze
    return SafetyResult(
        safe=False,
        level="dangerous",
        message=(
            f"Model needs ~{needs.ram_needed_gb}GB RAM. "
            f"You have {state.ram_available_gb}GB free.\n"
            f"This WILL freeze your laptop."
        ),
        ram_ok=False,
        gpu_ok=gpu_ok,
        suggestions=suggestions,
    )


def display_safety_check(result: SafetyResult) -> bool:
    """Show the safety check result. Returns True if should proceed."""
    if result.safe:
        console.print(f"[dim]{result.message}[/dim]")
        return True

    if result.level == "borderline":
        console.print(Panel(
            f"[yellow]Warning:[/yellow] {result.message}",
            title="Resource Check",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            f"[red bold]Warning:[/red bold] {result.message}",
            title="Resource Check",
            border_style="red",
        ))

    console.print("\n[bold]Options:[/bold]")
    for i, suggestion in enumerate(result.suggestions, 1):
        console.print(f"  {i}. {suggestion}")

    if result.level == "borderline":
        console.print("\n[dim]Proceeding anyway (use --force to skip this check).[/dim]")
        return True

    return False


# --- Runtime Resource Limiter ---

LOG_DIR = Path.home() / ".pixo" / "logs"


class ResourceLimiter:
    """Monitors and caps resource usage DURING model execution.

    Runs a background thread that checks RAM/CPU every 2 seconds.
    If usage gets dangerous, it triggers garbage collection or pauses.
    """

    def __init__(
        self,
        max_ram_percent: float = 70.0,
        max_cpu_cores: int | None = None,
        max_gpu_mem_gb: float | None = None,
    ):
        self.max_ram_percent = max_ram_percent
        self.max_cpu_cores = max_cpu_cores
        self.max_gpu_mem_gb = max_gpu_mem_gb

        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # start unpaused

        self.paused = False
        self._thermal_paused = False
        self.warnings: list[str] = []

        # Setup logging
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("pixo.guardian")
        handler = logging.FileHandler(LOG_DIR / "guardian.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        if not self._logger.handlers:
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def start(self):
        """Start the resource monitoring thread."""
        self._apply_cpu_affinity()
        self._apply_gpu_limit()

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        """Stop the monitoring thread."""
        self._stop_event.set()
        self._pause_event.set()  # unpause so thread can exit
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def wait_if_paused(self):
        """Call this in your processing loop. Blocks if guardian has paused execution."""
        self._pause_event.wait()

    def _apply_cpu_affinity(self):
        """Pin process to N-1 cores (leave 1 for OS)."""
        try:
            total_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
            max_cores = self.max_cpu_cores or max(total_cores - 1, 1)
            cores_to_use = list(range(min(max_cores, total_cores)))

            proc = psutil.Process()
            proc.cpu_affinity(cores_to_use)
            self._logger.info(f"CPU affinity set to cores {cores_to_use}")
        except Exception as e:
            self._logger.warning(f"Could not set CPU affinity: {e}")

    def _apply_gpu_limit(self):
        """Limit GPU memory if requested."""
        if self.max_gpu_mem_gb is None:
            return
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_mem
                fraction = min(self.max_gpu_mem_gb * (1024 ** 3) / total, 0.95)
                torch.cuda.set_per_process_memory_fraction(fraction)
                self._logger.info(f"GPU memory capped at {self.max_gpu_mem_gb}GB ({fraction:.0%})")
        except Exception as e:
            self._logger.warning(f"Could not set GPU memory limit: {e}")

    def _monitor_loop(self):
        """Background thread: check resources every 2 seconds."""
        while not self._stop_event.is_set():
            try:
                mem = psutil.virtual_memory()
                ram_percent = mem.percent
                ram_avail_gb = round(mem.available / (1024 ** 3), 1)

                self._logger.info(f"RAM: {ram_percent:.0f}% used, {ram_avail_gb}GB free")

                # Level 1: Warning at 80%
                if ram_percent > 80 and not self.paused:
                    self._logger.warning(f"RAM high: {ram_percent:.0f}%  -- forcing GC")
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

                # Level 2: Pause at 90%
                if ram_percent > 90 and not self.paused:
                    self.paused = True
                    self._pause_event.clear()
                    msg = f"RAM critical: {ram_percent:.0f}% -- pausing processing"
                    self._logger.warning(msg)
                    self.warnings.append(msg)
                    console.print(f"\n[yellow bold]Guardian:[/yellow bold] RAM at {ram_percent:.0f}%. Pausing to prevent freeze...")

                # Resume when RAM drops below 75%
                if ram_percent < 75 and self.paused and not self._thermal_paused:
                    self.paused = False
                    self._pause_event.set()
                    self._logger.info("RAM recovered, resuming")
                    console.print("[green]Guardian:[/green] RAM recovered. Resuming...")

                # Temperature monitoring
                temp = get_cpu_temperature()
                if temp is not None:
                    self._logger.info(f"CPU temp: {temp:.0f}C")

                    if temp > 95 and not self._thermal_paused:
                        # Critical: stop
                        self._thermal_paused = True
                        self.paused = True
                        self._pause_event.clear()
                        msg = f"CPU critical: {temp:.0f}C -- stopping to prevent damage"
                        self._logger.warning(msg)
                        self.warnings.append(msg)
                        console.print(f"\n[red bold]Guardian:[/red bold] CPU at {temp:.0f}C! Pausing until cooled down...")

                    elif temp > 85 and not self._thermal_paused:
                        # Hot: pause
                        self._thermal_paused = True
                        self.paused = True
                        self._pause_event.clear()
                        msg = f"CPU hot: {temp:.0f}C -- pausing to cool down"
                        self._logger.warning(msg)
                        console.print(f"\n[yellow bold]Guardian:[/yellow bold] CPU at {temp:.0f}C. Pausing to cool down...")

                    elif temp < 72 and self._thermal_paused:
                        # Cooled down: resume
                        self._thermal_paused = False
                        if ram_percent < 90:
                            self.paused = False
                            self._pause_event.set()
                            self._logger.info(f"CPU cooled to {temp:.0f}C, resuming")
                            console.print(f"[green]Guardian:[/green] CPU cooled to {temp:.0f}C. Resuming...")

            except Exception as e:
                self._logger.error(f"Monitor error: {e}")

            self._stop_event.wait(timeout=2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# --- Temperature monitoring ---

def get_cpu_temperature() -> float | None:
    """Try to read CPU temperature. Returns celsius or None if unavailable.

    Tries in order: psutil (Linux), WMI (Windows), fallback to None.
    Never crashes — temperature monitoring is optional.
    """
    # Try psutil (works on Linux)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if name in temps and temps[name]:
                    return temps[name][0].current
            # Try first available sensor
            for sensor_list in temps.values():
                if sensor_list:
                    return sensor_list[0].current
    except (AttributeError, Exception):
        pass

    # Try WMI (Windows)
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi "
             "| Select -ExpandProperty CurrentTemperature"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # WMI returns temperature in tenths of Kelvin
            raw = float(result.stdout.strip().split('\n')[0])
            celsius = (raw / 10.0) - 273.15
            if 0 < celsius < 120:  # sanity check
                return round(celsius, 1)
    except Exception:
        pass

    return None


# --- Execution modes ---

def apply_background_mode():
    """Set process to lowest priority so user can work normally."""
    import sys
    proc = psutil.Process()
    try:
        if sys.platform == "win32":
            import ctypes
            IDLE_PRIORITY_CLASS = 0x00000040
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), IDLE_PRIORITY_CLASS
            )
        else:
            import os
            os.nice(19)
        console.print("[dim]Background mode: process priority set to lowest.[/dim]")
    except Exception:
        # Fallback: use psutil
        try:
            proc.nice(psutil.IDLE_PRIORITY_CLASS if sys.platform == "win32" else 19)
        except Exception:
            pass


def apply_low_memory_cleanup():
    """Aggressive memory cleanup — call between frames in low-memory mode."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def suggest_modes():
    """Check current system state and suggest --low-memory or --background."""
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)
    cpu_percent = psutil.cpu_percent(interval=0.5)

    suggestions = []
    if avail_gb < 4:
        suggestions.append(
            f"Low RAM ({avail_gb:.1f}GB free). Consider: --low-memory"
        )
    if cpu_percent > 60:
        suggestions.append(
            f"CPU busy ({cpu_percent:.0f}% used). Consider: --background"
        )
    return suggestions
