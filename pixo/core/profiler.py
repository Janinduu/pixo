"""Hardware profiler — detects CPU, RAM, GPU and recommends optimization settings."""

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class HardwareProfile:
    cpu_name: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_name: str | None
    gpu_vram_gb: float | None
    cuda_version: str | None
    os_name: str
    disk_free_gb: float  # free space in ~/.pixo/

    @property
    def has_gpu(self) -> bool:
        return self.gpu_name is not None

    @property
    def recommendation(self) -> str:
        if self.has_gpu and self.gpu_vram_gb and self.gpu_vram_gb >= 4:
            return "Use GPU with TensorRT (fastest). Run: pixo optimize <model>"
        if self.has_gpu:
            return "Use GPU with ONNX Runtime. Run: pixo optimize <model>"
        if self.ram_total_gb >= 8:
            return "Use ONNX + INT8 quantization for best CPU speed. Run: pixo optimize <model>"
        return "Use ONNX + INT8 quantized + lite model variants for low memory. Try: pixo pull yolov8:small --optimize"


def _get_cpu_name() -> str:
    """Get a human-readable CPU name."""
    name = platform.processor()
    if not name or name == "":
        # fallback for some platforms
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            name = info.get("brand_raw", "Unknown CPU")
        except ImportError:
            name = platform.machine()
    return name


def _get_gpu_info() -> tuple[str | None, float | None, str | None]:
    """Detect GPU name, VRAM, and CUDA version. Returns (None, None, None) if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_gb = round(vram_bytes / (1024 ** 3), 1)
            cuda_version = torch.version.cuda
            return gpu_name, vram_gb, cuda_version
    except ImportError:
        pass
    return None, None, None


def get_profile() -> HardwareProfile:
    """Detect hardware and return a HardwareProfile."""
    gpu_name, gpu_vram, cuda_version = _get_gpu_info()

    pixo_dir = Path.home() / ".pixo"
    disk_free = shutil.disk_usage(pixo_dir if pixo_dir.exists() else Path.home()).free
    disk_free_gb = round(disk_free / (1024 ** 3), 1)

    mem = psutil.virtual_memory()

    return HardwareProfile(
        cpu_name=_get_cpu_name(),
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count(),
        ram_total_gb=round(mem.total / (1024 ** 3), 1),
        ram_available_gb=round(mem.available / (1024 ** 3), 1),
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        cuda_version=cuda_version,
        os_name=f"{platform.system()} {platform.release()}",
        disk_free_gb=disk_free_gb,
    )
