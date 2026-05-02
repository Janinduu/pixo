# Changelog

All notable changes to pixo will be documented in this file.

---

## v0.3.3 (2026-05-03)

**Real airgap fix for HuggingFace transformers models.**

Discovered while testing v0.3.2 that `--airgap` was still leaking on `depth_anything_v2`, `grounding_dino`, `florence2`, `sam2`, and `samurai`. Root cause: `huggingface_hub` reads `HF_HUB_OFFLINE` once at module import time (in its `constants.py`). Pixo's CLI was importing `huggingface_hub` transitively (via `pixo.core.downloader`) before the airgap context could set the env var, so the offline mode was a no-op for HF-backed runners.

### Fix
- **Set offline env vars at the very top of `pixo/cli.py`** — before any `huggingface_hub` import. If `--airgap` is anywhere in `sys.argv`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, and `YOLO_OFFLINE` are all set immediately. This means transformers picks up offline mode from the start of the process, not just inside the airgap context.
- **New pre-flight check** for HF transformers models: even when the model's weight file exists, the model's processor/config might not be in the HF cache yet. If `--airgap` is requested and the HF cache directory doesn't contain the model, refuse with a friendly message ("Run once without --airgap so transformers can populate its cache").

---

## v0.3.2 (2026-05-02)

**Stress-test patch — surfaced and fixed a batch of real-world friction points found while exercising v0.3 end-to-end.**

### Routing and pre-flight checks
- **Video + cloud:** the Kaggle/Colab backends in v0.3 only support image inputs. Pixo now refuses video-to-cloud routing pre-flight with a clear message and a `--backend local` suggestion, instead of silently failing inside the kernel script.
- **Smart router fallback:** when the router auto-recommends cloud for a video input, it now falls back to local with a notice, rather than queueing a doomed cloud run.
- **Model + input mismatch:** running `sam2` on a video now fails fast and points to `samurai` (designed for video). Same pattern for other image-only models.
- **Airgap + un-pulled weights:** running with `--airgap` on a model that isn't downloaded yet now fails fast with a friendly message ("run once without --airgap to fetch weights"), instead of dumping a 200-line Ultralytics traceback.

### Cloud history and sharing
- **Cloud runs in `pixo history`:** runs routed to Kaggle/Colab are now tracked as checkpointed jobs, so they show up in `pixo history`, can be opened with `pixo view`, and can be exported with `pixo share`.
- **`pixo share` error message:** when an old job's output directory has been deleted, `pixo share` now explains *why* (with the expected path and date) instead of just "output directory not found".

### Quality and packaging
- **Logo on PyPI:** the README's logo path is now an absolute GitHub URL, so the PyPI project page renders it correctly.
- **Kaggle dependency:** pinned to `kaggle>=2.0.2` to silence the outdated-version warning that was appearing on every cloud run.
- **Kernel log on Windows:** suppressed the misleading "could not save kernel log (encoding issue)" message — the actual results were always downloading fine; only the optional log was affected.

---

## v0.3.1 (2026-04-24)

**Patch release.**

- Add `pixo --version` / `pixo -V` flag to print the installed version and exit.

---

## v0.3.0 (2026-04-23)

**The hero-moment release — zero-setup demo, shareable reports, airgap mode, and cross-model comparison.**

### `pixo try` — zero-argument demo
One command takes a new user from install to a working result in under a minute.
- Auto-picks a model based on the user's hardware (GPU → yolov11, CPU → yolov8).
- Finds a sample image (bundled with ultralytics, or cached in `~/.pixo/samples/`).
- Auto-pulls the model if needed, runs it, opens a browser report.
- Usage: `pixo try` (no flags needed). Override with `--model` or `--input`.

### `pixo share` — self-contained HTML reports
Produce a single `.html` file with results, visualizations, and provenance baked in as base64. Opens in any browser with zero network requests.
- `pixo share` shares the most recent completed run.
- `pixo share <job_id>` shares a specific job.
- Output: `~/.pixo/shares/<job_id>.html` — attach to a tweet, Slack, or email.

### `pixo compare` — disagreement browser
Run multiple detection models on one image and see only where they disagree.
- `pixo compare yolov8 yolov11 yolov12 --input photo.jpg`
- Classifies each detection as agreement (all models found it), partial (some did), or unique (only one did).
- Produces a standalone HTML report with colored overlays per model.
- v0.3 scope: image inputs and Ultralytics detection models (yolov8/11/12, rtdetr). Video + segmentation compare planned for v0.4.

### `--airgap` mode
Hard-block all outbound network calls during a run.
- `pixo run yolov8 --input photo.jpg --airgap`
- Monkey-patches `socket.connect` and `getaddrinfo` to block non-loopback traffic.
- Raises `AirgapViolation` if any code tries to reach the internet.
- Incompatible with `--backend kaggle/colab` (by design).

### Privacy badges on model cards
Each model card now declares its privacy posture:
- **green** — runs fully offline after initial weight download
- **yellow** — needs network for first pull, offline afterward
- **red** — requires runtime network access (API calls, remote services)
- Surfaced in `pixo list` and `pixo info`.
- All nine bundled models are **green**.

### `pixo serve` — instant browser UI
One command spins up a Gradio UI for any model.
- `pixo serve yolov8` → http://localhost:7860
- Drag-drop an image, see annotated result and summary.
- Auto-shows a prompt field for grounding_dino and task dropdown for florence2.
- Requires the new `pixo[demo]` extra: `pip install pixo[demo]`.

### Other
- New core modules: `pixo/core/sample.py`, `pixo/core/share.py`, `pixo/core/airgap.py`, `pixo/core/compare.py`.
- `pyproject.toml`: new `demo` extra for Gradio.
- Model card schema: new optional `privacy:` section.

---

## v0.2.0 (2026-04-08)

**All models working + isolated environments + model piping.**

### All 6 Model Runners Now Working
- **GroundingDINO** — text-prompted object detection (`--prompt "person, car"`)
  - Uses `transformers` zero-shot-object-detection pipeline
  - Supports image and video input
  - Draws bounding boxes with labels and confidence scores
  - Saves detections.json with all detection data
- **Florence-2** — vision-language model with multiple tasks
  - Caption: `--task caption` or `--task detailed_caption`
  - Detection: `--task detect` (bounding boxes)
  - OCR: `--task ocr` (text extraction)
  - Saves structured JSON results per task
- **SAMURAI** — video object tracking + segmentation
  - Uses SAM2 mask generation for per-frame segmentation
  - Consistent color palette across video frames
  - Saves overlay video + individual masks per frame

### Isolated Environments (opt-in)
- `pixo pull <model> --isolate` creates a per-model venv at `~/.pixo/envs/<model>/`
- Automatically installs dependencies from modelcard.yaml
- `pixo run <model> --isolate` runs in the model's own venv
- `pixo env-list` shows all environments with sizes
- `pixo env-clean <model>` removes an environment
- Default: models still run in user's current environment (no breaking change)

### Model Piping
- `pixo pipe "grounding_dino -> sam2" --input photo.jpg --prompt "person"`
- Chain any models together with `->` or `→` separator
- Pre-built templates: `detect_and_segment`, `detect_and_track`, `segment_and_depth`
- Automatic output-to-input conversion between compatible models
- Step-by-step progress display

### Model Card Updates
- GroundingDINO: default changed to `grounding-dino-tiny` (smaller, faster)
- Added `base` variant for GroundingDINO

---

## v0.1.1 (2026-04-08)

**Patch release** — cleanup and metadata fixes.

- Removed internal development files from repo
- Fixed project URLs pointing to correct GitHub repository
- Updated LICENSE with author information
- Cleaned up README (removed roadmap section)

---

## v0.1.0 (2026-04-08)

**Initial release** — the foundation of pixo.

### Core Engine
- CLI with `pull`, `run`, `list`, `info`, `doctor`, `rm` commands
- Model registry with YAML-based model cards
- HuggingFace model downloader with variant support (`pixo pull sam2:tiny`)
- Auto-pull: if a model isn't downloaded, `pixo run` downloads it automatically

### Working Models
- **YOLOv8** — real-time object detection (images + video), fully tested
- **SAM2** — segment anything (auto-mask generation), working runner
- **Depth Anything V2** — monocular depth estimation (colored/grayscale/raw output), working runner
- **SAMURAI** — video tracking + segmentation (model card ready, stub runner)
- **GroundingDINO** — open-set text-prompted detection (model card ready, stub runner)
- **Florence-2** — vision-language model (model card ready, stub runner)

### Resource Guardian (Phase 4)
- Pre-run safety check: estimates RAM needs, blocks dangerous runs, warns on borderline
- Runtime monitoring: background thread checks RAM/CPU every 2 seconds
- Auto-pause at 90% RAM usage, auto-resume when recovered
- Temperature monitoring: pauses at 85°C, stops at 95°C, resumes at 72°C
- `--low-memory` mode: frame-by-frame processing with aggressive garbage collection
- `--background` mode: lowest OS priority, fewer cores, laptop stays fully usable
- Auto-suggestions when system resources are tight

### ONNX Optimization (Phase 2)
- `pixo optimize <model>` converts PyTorch to ONNX format
- 40% CPU speedup with zero configuration
- Auto-detects optimized model and uses it

### Cloud GPU Routing (Phase 3)
- Kaggle backend: fully automated (upload → run on GPU → download results)
- Colab backend: semi-automated (generates notebook, user clicks Run)
- Smart router: estimates time per backend, shows comparison table
- `pixo setup-cloud` for one-time account connection
- `--backend kaggle` or `--backend colab` to force a specific backend

### Checkpointing (Phase 6)
- Auto-saves progress every N frames to `~/.pixo/checkpoints/`
- `pixo resume` picks up from last checkpoint
- First Ctrl+C pauses gracefully (saves checkpoint), second Ctrl+C quits
- `pixo history` shows all jobs with status and progress
- `pixo jobs-clean` removes completed job checkpoints
- Deterministic job IDs: same model + input always maps to same job
- Integration with Resource Guardian: auto-saves checkpoint on pause/crash

### Standard Output (Phase 7.1)
- Every run produces consistent output structure:
  - `results.json` — machine-readable metadata (model, timing, device, results)
  - `summary.txt` — human-readable one-pager
  - `visualizations/` — annotated images or video
  - `raw/` — model-specific raw outputs
  - `exports/` — COCO JSON + CSV exports
- `pixo view <job_id>` opens output folder

### Plugin System (Phase 5)
- Models defined by `modelcard.yaml` + `run.py` (two files per model)
- `setup()` and `run()` interface — 20-50 lines per model
- Core is torch-free: base runner doesn't import torch
- `pixo info <model>` shows runner status (ready vs stub)
- Old hardcoded runner system completely replaced

### Hardware Profiler
- `pixo doctor` shows CPU, RAM, GPU, disk, temperature, recommendations

---

## v0.2.0 (upcoming)

### Planned Features

- **All 6 model runners working** — SAMURAI, GroundingDINO, and Florence-2 runners implemented (not just stubs)
- **Isolated environments** — per-model Python venvs so dependencies never conflict (`~/.pixo/envs/<model>/`)
- **Model piping** — chain models: `pixo pipe "grounding_dino → sam2 → samurai" --input video.mp4`
- **Improved cloud routing** — better error handling, session management, auto-retry on disconnect

---

## v0.3.0 (future)

### Planned Features

- **Web dashboard** — `pixo ui` opens a local web interface (optional: `pip install pixo[web]`)
- **FastAPI backend** — REST + WebSocket API for live progress
- **React frontend** — model grid, drag-and-drop upload, live resource graphs, video results viewer

---

## v0.4.0+ (future)

### Planned Features

- **pixo Hub** — browse models online, see benchmarks, community ratings
- **Community model registry** — separate repo, anyone submits models via PR
- **Benchmark database** — users submit run times to help estimate for others
- **Pipeline templates** — common workflows as one-click configs
- **Pre-built environments** — download ready-made venvs for popular models
