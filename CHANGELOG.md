# Changelog

All notable changes to pixo will be documented in this file.

---

## v0.1.1 (2026-04-08)

**Patch release** — cleanup and metadata fixes.

- Removed internal development files from repo
- Fixed project URLs pointing to correct GitHub repository
- Updated LICENSE with author information
- Cleaned up README (removed roadmap section)

---

## v0.1.0 (2026-04-08)

**Initial release** — the foundation of "Ollama for Computer Vision."

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
