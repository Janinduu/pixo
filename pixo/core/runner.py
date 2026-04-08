"""Base model runner — utility functions for model execution."""


def get_device(force: str | None = None) -> str:
    """Auto-detect best available device. Lazy torch import."""
    if force:
        return force
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
