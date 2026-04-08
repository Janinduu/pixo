# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pixo is "Ollama for Computer Vision" — a free, open-source CLI tool and platform that lets anyone run **any** heavy CV model on **any** laptop without freezing your machine, losing progress, or fighting dependency hell.

The six core problems pixo solves:
1. **Laptop freezing** — Resource Guardian caps RAM/CPU/GPU, never lets the machine lock up
2. **Lost progress** — Checkpointing saves every N frames, Ctrl+C pauses instead of killing
3. **Dependency conflicts** — Isolated venvs per model (v0.2, opt-in)
4. **No estimates** — Smart router shows time estimates before running, picks fastest backend
5. **Inconsistent outputs** — Standard output format (results.json, visualizations, COCO export)
6. **Setup hell** — One command: `pixo pull <model> && pixo run <model> --input file`

## Tech Stack

- **CLI**: Python (Typer), Rich for terminal UI
- **Resource monitoring**: psutil, GPUtil (optional)
- **Model optimization**: ONNX Runtime, TensorRT (auto-detected)
- **Cloud backends**: Kaggle API (fully automated), Google Colab (semi-automated)
- **Local server**: FastAPI with WebSocket (optional: `pip install pixo[web]`)
- **Web dashboard**: React + Vite + Tailwind CSS (optional)
- **Model definitions**: modelcard.yaml + run.py per model (plugin system)
- **Package**: pip-installable (`pip install pixo`), uses pyproject.toml

## Build & Development Commands

```bash
pip install -e .              # Install in dev mode
pixo pull <model>             # Download a model
pixo run <model> --input <file>  # Run inference
pixo list                     # List available models
pixo doctor                   # Check hardware
pixo optimize <model>         # Convert to ONNX for faster CPU inference
pixo setup-cloud              # Connect Kaggle/Colab accounts
pixo cloud-status             # Check cloud backend status
```

## Architecture

### Completed (Phases 1-5)

- **Phase 1 (Core)**: CLI skeleton, model registry, HuggingFace downloader, YOLOv8 runner
- **Phase 2 (Optimization)**: Hardware profiler (`pixo doctor`), ONNX auto-optimization (40% CPU speedup)
- **Phase 3 (Cloud)**: Kaggle backend (fully automated), Colab backend (notebook generation), smart router with time estimates
- **Phase 4 (Resource Guardian)**: Pre-run safety checks, runtime resource limits, low-memory mode, background mode, temperature monitoring
- **Phase 5 (Plugin System)**: `core/plugin.py` with ModelCard + PluginLoader, 6 model cards, 3 working runners (yolov8, sam2, depth_anything_v2), 3 stubs (samurai, grounding_dino, florence2). Old hardcoded system deleted. Core is torch-free.
- **Phase 6 (Checkpointing)**: `core/checkpoint.py` with CheckpointManager, auto-save every N frames, auto-resume on re-run, `pixo history`/`pixo resume`/`pixo jobs clean`, Ctrl+C pauses gracefully (second Ctrl+C force-quits), deterministic job IDs, error recovery.
- **Phase 7.1 (Standard Output)**: `core/output.py` with OutputFormatter, unified output structure (results.json, summary.txt, visualizations/, raw/, exports/), COCO export, CSV export, `pixo view` command.
- **Phase 5.2 (Isolated Envs)**: `core/envmanager.py`, per-model venvs via `--isolate` flag, `pixo env-list`, `pixo env-clean`.
- **Phase 7.2 (Model Piping)**: `core/pipeline.py`, `pixo pipe "grounding_dino -> sam2"`, pre-built templates, automatic converters.
- **All 6 runners working**: GroundingDINO (text-prompted detection), Florence-2 (captioning/detection/OCR), SAMURAI (video tracking via SAM2).

### Post-Launch (v0.3+)

- **Phase 8 (Web Dashboard)**: Optional install, FastAPI + React

### Key Design Patterns

- **Plugin system**: Models defined by `modelcard.yaml` + `run.py` in `pixo/models/cards/<model>/`
- **Runner interface**: `setup(model_dir, variant, device)` and `run(model, input_path, output_dir, options)` — that's it
- **Core is torch-free**: Base runner.py does not import torch. Each model's run.py imports what it needs.
- **Smart router** (`pixo/cloud/router.py`) estimates local vs cloud time using actual frame counts
- **Model variants** use colon syntax: `pixo pull sam2:lite`, `pixo pull sam2:tiny`
- **Resource Guardian** (`pixo/core/guardian.py`) wraps model execution with RAM/CPU/GPU/temp limits
- Models stored in `~/.pixo/models/`, config in `~/.pixo/config.yaml`, logs in `~/.pixo/logs/`

## Development Approach

Build prompts are in `files/prompts.md` — run them in order. Each prompt builds on the previous. Test after each step before moving on. The plan is in `files/plan.md`.
