"""Self-contained HTML report generator for `pixo share`.

Takes an OutputFormatter-produced run directory and produces a single .html
file with all images inlined as base64 data URIs. The resulting file opens
in any browser with no network requests and no external dependencies.
"""

import base64
import html
import json
import mimetypes
import time
from pathlib import Path

from pixo import __version__

SHARE_DIR = Path.home() / ".pixo" / "shares"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_VIDEO_EXTS = {".mp4", ".webm", ".mov"}


def _mime(path: Path) -> str:
    m, _ = mimetypes.guess_type(str(path))
    return m or "application/octet-stream"


def _data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime(path)};base64,{data}"


def _find_run_dir(job_id: str) -> Path | None:
    """Find the structured output dir for a given job_id prefix.

    Looks in common locations: ./pixo_output, and the user-provided output
    directory stored in the checkpoint.
    """
    from pixo.core.checkpoint import CheckpointManager

    mgr = CheckpointManager()
    matches = [j for j in mgr.list_jobs() if j.job_id.startswith(job_id)]
    if not matches:
        return None

    job = matches[0]
    output_dir = Path(job.output_dir)
    if not output_dir.exists():
        return None

    prefix = job.job_id[:8]
    for sub in output_dir.iterdir():
        if sub.is_dir() and sub.name.startswith(prefix):
            return sub
    return None


def build_html_report(run_dir: Path) -> str:
    """Render a run directory as a single self-contained HTML string."""
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    results = json.loads(results_path.read_text())

    viz_dir = run_dir / "visualizations"
    media_blocks = []
    if viz_dir.exists():
        files = sorted(viz_dir.iterdir())
        # Limit to ~40 images or 3 videos to keep file size reasonable
        image_files = [f for f in files if f.suffix.lower() in _IMAGE_EXTS][:40]
        video_files = [f for f in files if f.suffix.lower() in _VIDEO_EXTS][:3]

        for f in video_files:
            media_blocks.append(
                f'<video controls src="{_data_uri(f)}" '
                f'style="max-width:100%;border-radius:8px;margin-bottom:16px"></video>'
            )
        for f in image_files:
            media_blocks.append(
                f'<figure><img src="{_data_uri(f)}" alt="{html.escape(f.name)}"/>'
                f'<figcaption>{html.escape(f.name)}</figcaption></figure>'
            )

    # Summary fields
    model = html.escape(str(results.get("model", "?")))
    variant = html.escape(str(results.get("variant", "")))
    device = html.escape(str(results.get("device", "?")))
    timing = results.get("timing", {}) or {}
    total_secs = timing.get("total_seconds", 0)
    fps = timing.get("fps", 0)
    summary = results.get("results_summary", {}) or {}
    input_info = results.get("input", {}) or {}
    input_name = html.escape(Path(str(input_info.get("path", "?"))).name)

    summary_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
    )

    job_id = html.escape(str(results.get("job_id", "")))
    pixo_v = html.escape(str(results.get("pixo_version", __version__)))
    generated = time.strftime("%Y-%m-%d %H:%M")

    media_html = "\n".join(media_blocks) if media_blocks else \
        "<p class='empty'>No visualizations were produced.</p>"

    return _TEMPLATE.format(
        title=f"pixo {model} — {input_name}",
        model=model,
        variant=variant or "default",
        device=device,
        input_name=input_name,
        total_secs=f"{total_secs}s",
        fps=f"{fps} fps" if fps else "—",
        summary_rows=summary_rows or "<tr><td colspan='2' class='empty'>No summary fields.</td></tr>",
        media_html=media_html,
        job_id=job_id,
        pixo_v=pixo_v,
        generated=generated,
    )


def create_share_bundle(run_dir: Path, out_path: Path | None = None) -> Path:
    """Write a self-contained HTML file for this run. Returns the file path."""
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        job_id = run_dir.name.split("_", 1)[0]
        out_path = SHARE_DIR / f"{job_id}.html"
    out_path.write_text(build_html_report(run_dir), encoding="utf-8")
    return out_path


def find_run_dir_by_job(job_id: str) -> Path | None:
    """Public helper — find the run directory for a job id prefix."""
    return _find_run_dir(job_id)


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 960px; margin: 0 auto; padding: 32px 24px;
    background: #f8fafc; color: #0f172a;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0f172a; color: #e2e8f0; }}
    .card {{ background: #1e293b; border-color: #334155; }}
    th, td {{ border-color: #334155; }}
  }}
  header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:24px; }}
  h1 {{ margin:0; font-size:22px; font-weight:600; }}
  .sub {{ color:#64748b; font-size:13px; }}
  .badge {{
    display:inline-block; padding:2px 8px; border-radius:999px;
    background:#0ea5e9; color:white; font-size:11px; font-weight:600;
    letter-spacing:0.03em; text-transform:uppercase;
  }}
  .card {{
    background:white; border:1px solid #e2e8f0; border-radius:12px;
    padding:20px; margin-bottom:20px;
  }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:16px; }}
  .stat {{ }}
  .stat .label {{ font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }}
  .stat .value {{ font-size:18px; font-weight:600; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e2e8f0; }}
  th {{ font-weight:500; color:#64748b; width:40%; }}
  figure {{ margin:0 0 16px; }}
  figure img {{ max-width:100%; border-radius:8px; display:block; }}
  figcaption {{ font-size:12px; color:#64748b; margin-top:4px; }}
  .empty {{ color:#94a3b8; font-style:italic; text-align:center; padding:24px; }}
  footer {{ margin-top:32px; font-size:12px; color:#64748b; text-align:center; }}
  footer a {{ color:#0ea5e9; text-decoration:none; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>{model} <span class="sub">on {input_name}</span></h1>
    <div class="sub">variant: {variant} · device: {device}</div>
  </div>
  <span class="badge">pixo {pixo_v}</span>
</header>

<div class="card">
  <div class="grid">
    <div class="stat"><div class="label">Time</div><div class="value">{total_secs}</div></div>
    <div class="stat"><div class="label">Speed</div><div class="value">{fps}</div></div>
    <div class="stat"><div class="label">Device</div><div class="value">{device}</div></div>
    <div class="stat"><div class="label">Job</div><div class="value" style="font-family:monospace">{job_id}</div></div>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0;font-size:16px">Summary</h2>
  <table>{summary_rows}</table>
</div>

<div class="card">
  <h2 style="margin-top:0;font-size:16px">Visualizations</h2>
  {media_html}
</div>

<footer>
  Generated {generated} with <a href="https://github.com/Janinduu/pixo">pixo</a> — local-first computer vision.
  <br/>This file is fully self-contained. No network calls were made to render it.
</footer>

</body>
</html>
"""
