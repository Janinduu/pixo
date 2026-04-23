"""Sample input discovery for `pixo try`.

Locates a demo image so new users can see pixo work with zero setup.
Tries, in order:
  1. ultralytics' bundled bus.jpg (no network, if ultralytics is installed)
  2. cached copy in ~/.pixo/samples/
  3. downloaded from ultralytics' public GitHub assets (one-time, ~130 KB)
"""

from pathlib import Path

SAMPLE_DIR = Path.home() / ".pixo" / "samples"
SAMPLE_NAME = "bus.jpg"
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"


def _ultralytics_bundled_sample() -> Path | None:
    """Return path to ultralytics' bundled bus.jpg if available, else None."""
    try:
        import ultralytics
    except ImportError:
        return None
    assets = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    return assets if assets.exists() else None


def _cached_sample() -> Path | None:
    path = SAMPLE_DIR / SAMPLE_NAME
    return path if path.exists() else None


def _download_sample() -> Path:
    """Download the sample image to ~/.pixo/samples/. One-time network call."""
    import requests
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    target = SAMPLE_DIR / SAMPLE_NAME
    resp = requests.get(SAMPLE_URL, timeout=15)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    return target


def get_sample_image(airgap: bool = False) -> Path | None:
    """Return a path to a sample image, or None if unavailable.

    If airgap is True, will never attempt to download.
    """
    bundled = _ultralytics_bundled_sample()
    if bundled:
        return bundled

    cached = _cached_sample()
    if cached:
        return cached

    if airgap:
        return None

    try:
        return _download_sample()
    except Exception:
        return None
