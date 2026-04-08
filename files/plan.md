# pixo — Project Plan (v3)

## Overview

Build an open-source CLI tool and platform that lets anyone run **any** heavy CV model
on any laptop — without freezing your machine, losing progress, or fighting
dependency hell. Follow Ollama's philosophy: simple commands, zero config.

Cost to the user: **$0**. Always.

---

## What Changed From v2

Phases 1-4 are built and working. This updated plan reflects what we learned:

- **Ship a usable v0.1 faster** — moved isolated envs and piping to post-launch
- **Migration matters** — Phase 5 now explicitly replaces the old config system (no two systems living side by side)
- **Core stays lightweight** — runner.py won't hard-depend on torch; each model imports what it needs
- **Web dashboard is optional** — install with `pip install pixo[web]`, not bundled in the core package
- **v0.1 release milestone** added after Phase 7.1 — that's when we ship to GitHub/PyPI

---

## Phases 1-4 — DONE

### Phase 1 — Core Engine (complete)
- CLI skeleton, model registry, downloader, YOLOv8 runner
- `pixo pull yolov8 && pixo run yolov8 --input video.mp4` works end-to-end
- 9582-frame video: 22 minutes on CPU, detected 20,217 objects

### Phase 2 — Local Optimization (complete)
- Hardware profiler (`pixo doctor`)
- ONNX auto-optimization: 40% speedup on CPU for free
- 22 minutes → ~13 minutes just by changing model format

### Phase 3 — Free Cloud GPUs (complete)
- Kaggle backend: fully automated (upload → run on GPU → download results)
- Colab backend: semi-automated (generates notebook, user clicks Run)
- Smart router: estimates time per backend, picks fastest

### Phase 4 — Resource Guardian (complete)
- Pre-run safety check: estimates RAM needs, blocks if dangerous, warns if borderline
- Runtime limiter: background thread monitors RAM/CPU every 2s, GC at 80%, pause at 90%
- Low-memory mode (`--low-memory`): frame-by-frame processing with aggressive GC
- Background mode (`--background`): lowest OS priority, fewer cores, 50% RAM cap
- Temperature monitoring: reads CPU temp, pauses at 85°C, stops at 95°C, resumes at 72°C
- Auto-suggestions when system resources are tight

---

## Phase 5 — Plugin System (Week 1-2)

**The Problem:**
Right now, adding a new model means writing a custom Python runner and
hardcoding it in `RUNNERS = {"yolov8": "pixo.models.runners.yolo_runner.YOLORunner"}`
in cli.py. That doesn't scale. If someone wants to run SAMURAI, Depth Anything V2,
or Florence-2, they shouldn't have to wait for us to code a runner.

**Goal:** Anyone can add a new model by writing a YAML file and a
short Python script. Just like Ollama's Modelfile.

### What to build

1. **Model card format (modelcard.yaml)**
   Every model is defined by one YAML file:
   ```yaml
   name: samurai
   description: Segment and track anything in video
   version: 1.0.0
   task: video-tracking-segmentation
   author: yangchris11

   source:
     type: github
     repo: yangchris11/samurai
     branch: main

   dependencies:
     python: ">=3.9,<3.12"
     packages:
       - sam2
       - numpy>=1.24
       - opencv-python>=4.8
       - torch>=2.0

   hardware:
     min_ram_gb: 8
     recommended_ram_gb: 16
     min_vram_gb: 4
     recommended_vram_gb: 8
     cpu_fallback: true

   inputs:
     - type: video
       formats: [mp4, avi, mov]
     - type: image
       formats: [jpg, png, bmp]

   outputs:
     - type: masks
       format: png
     - type: tracked_video
       format: mp4
     - type: metadata
       format: json

   variants:
     full:
       size_mb: 2500
       description: Best quality, needs GPU
     lite:
       size_mb: 800
       description: Good balance of speed and quality
     tiny:
       size_mb: 300
       description: Fastest, lower accuracy

   checkpoint:
     supported: true
     every: 100
   ```

2. **Runner script (run.py)**
   Each model has a small Python file with just two functions:
   ```python
   def setup(model_dir, variant, device):
       """Load the model. Called once."""
       return model

   def run(model, input_path, output_dir, options):
       """Run inference. Called per input file."""
       # yield progress updates: {"frame": 100, "total": 9582}
   ```

   That's it. 20-50 lines per model. pixo handles everything else.

3. **Migration from old system**
   - Replace `ModelConfig` dataclass in registry.py with new `ModelCard` format
   - Delete the hardcoded `RUNNERS` dict and `_get_runner()` function from cli.py
   - Migrate existing 3 YAML configs (yolov8, sam2, grounding_dino) to modelcard.yaml format
   - Move from `pixo/models/configs/*.yaml` to `pixo/models/cards/<model>/modelcard.yaml`
   - Make runner.py base class torch-free (lazy import for `get_device()`)
   - One config system, not two

4. **Model discovery**
   ```bash
   $ pixo search tracking
   Available models:
     samurai         — Segment and track anything in video
     sam2            — Meta's Segment Anything 2
     xmem            — Long-term video object segmentation
     cutie           — Putting the object back into segmentation
   ```

### What success looks like

```bash
$ pixo pull samurai
Downloading weights... done

$ pixo run samurai --input video.mp4
# Works via the plugin system, no hardcoded runner needed
```

No custom code. No dependency hell. Just pull and run.

---

## Phase 5.3 — Working Runners (Week 2)

Implement actual working run.py files for the most important models.
These cover detection, segmentation, and depth — the three most common CV tasks.

1. **yolov8** — port existing YOLORunner to the new setup()/run() plugin format
2. **sam2** — auto-mask generation + point/box prompts for images, frame-by-frame for video
3. **depth_anything_v2** — depth map generation (colored visualization + raw values + grayscale)

Each runner: under 80 lines, handles image + video, yields progress dicts.

---

## Phase 6 — Checkpointing & Recovery (Week 3-4)

**The Problem:**
You run SAMURAI on a 10-minute video. 8000 frames done out of 9582.
Then: Colab disconnects. Or laptop overheats. Or power goes out.
You start over from frame 1. Hours wasted.

**Goal:** Never lose progress. Ever.

### What to build

1. **Automatic checkpointing**
   - Every N frames (default 100), save:
     - Which frames are processed
     - Model state (embeddings, tracking IDs)
     - Partial output files
   - Stored in `~/.pixo/checkpoints/<job_id>/`

2. **Auto-resume**
   ```
   $ pixo run samurai --input video.mp4

   Found checkpoint: 8000/9582 frames complete (83%)
   Resume from checkpoint? [Y/n]

   Resuming from frame 8001...
   Took: 4m (saved ~38m from checkpoint)
   ```

3. **Manual pause/resume**
   - First Ctrl+C: pause (save checkpoint), don't kill
   - Second Ctrl+C within 3 seconds: actually quit
   - Press 'p' to pause, 'q' to save & quit, 's' for stats
   - Resume later: `pixo resume`

4. **Job history**
   ```bash
   $ pixo history
   ID   Model     Input          Status     Time     Date
   12   samurai   video.mp4      done       42m      2 hours ago
   11   yolov8    photos/        done       8m       yesterday
   10   samurai   long_vid.mp4   paused     3h/5h    yesterday

   $ pixo resume 10
   Resuming job #10 (60% complete)...
   ```

5. **Integration with ResourceGuardian**
   - When guardian pauses due to high RAM/temperature → auto-save checkpoint
   - When guardian kills job due to critical temperature → save checkpoint first
   - Message: "Saved checkpoint at frame 8000. Run `pixo resume` when ready."

---

## Phase 7.1 — Standard Output (Week 5)

**The Problem:**
Every model dumps results differently. No consistent format for downstream tools.

**Goal:** Every model outputs in the same format.

### What to build

1. **Unified output structure**
   ```
   pixo_output/
   └── job_12_samurai_video/
       ├── results.json          # metadata: model, settings, timing
       ├── summary.txt           # human-readable summary
       ├── visualizations/       # annotated images/video
       ├── raw/                  # machine-readable outputs
       │   ├── masks/
       │   ├── boxes/
       │   ├── labels/
       │   └── tracks/
       └── exports/
           ├── coco.json         # COCO format
           └── csv/
   ```

2. **results.json schema** with pixo version, job ID, model info, device, timing, resource usage, results summary

3. **Export commands:**
   - `pixo view <job_id>` — opens visualization in default viewer
   - COCO format export for interop with other CV tools
   - CSV export for spreadsheet workflows

4. **Update ALL model runners** to use OutputFormatter instead of custom output logic.

---

## ===== v0.1.0 RELEASE =====

**Ship it here.** At this point pixo has:
- ✅ Pull and run any supported model with one command
- ✅ Resource Guardian prevents laptop freezing
- ✅ Plugin system for easy model addition
- ✅ 3+ working models (yolov8, sam2, depth_anything_v2)
- ✅ Checkpointing (never lose progress)
- ✅ Pause/resume with Ctrl+C
- ✅ Standard output format
- ✅ Cloud GPU routing (Kaggle/Colab)
- ✅ ONNX optimization

**Release checklist:**
- Clean up README with installation, quickstart, supported models, GIFs
- Add LICENSE (MIT)
- Publish to PyPI: `pip install pixo`
- Create GitHub releases with changelog
- Add `pixo upgrade` command (runs `pip install --upgrade pixo`)

---

## Post-Launch Phases

### Phase 5.2 — Isolated Environments (v0.2)

Per-model virtual environment isolation. Deferred because:
- Adds subprocess complexity (progress streaming, error forwarding)
- Windows edge cases are significant
- Most users running 1-2 models won't hit conflicts yet

**What to build:**
- `create_env(model_name)` → creates venv at `~/.pixo/envs/{model_name}/`
- `pixo pull` creates venv + installs deps from modelcard.yaml
- `pixo run` executes model's run.py using the model's own venv python
- `pixo env list` / `pixo env clean` commands
- `--no-env` flag for advanced users who manage their own environment
- Default: models install into user's current environment
- `--isolate` flag opts into per-model venv

### Phase 7.2 — Model Piping (v0.2)

Chain models together: `pixo pipe "grounding_dino → sam2 → samurai"`

**What to build:**
- Standard intermediate formats (detection → segmentation → tracking)
- Converter functions between model outputs/inputs
- Pre-built pipeline templates as YAML files
- Step-by-step progress display

### Phase 8 — Web Dashboard (v0.3)

**Installed separately:** `pip install pixo[web]`

**8.1 — FastAPI backend:**
- REST endpoints + WebSocket for live progress
- Job management, model browsing, file upload
- `pixo ui` command starts server + opens browser

**8.2 — React dashboard:**
- Model grid with search/filter
- Drag-and-drop run page with live resource graphs
- Video results viewer with frame scrubber
- Jobs page with pause/resume
- Settings page for cloud connections + resource limits

### Phase 9 — Community & Growth (v0.4+)

1. **pixo Hub** — browse models, see benchmarks, community ratings
2. **Benchmark database** — users submit run times, helps estimate for others
3. **Pre-built environments** — download ready-made venvs for popular models
4. **Pipeline templates** — common workflows as one-click configs
5. **Community model registry** — separate `pixo-models/registry` repo, anyone submits PRs

---

## Updated File Structure

```
pixo/
├── pyproject.toml
├── README.md
├── LICENSE
├── pixo/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI commands
│   ├── core/
│   │   ├── runner.py           # Base runner (torch-free)
│   │   ├── plugin.py           # Plugin loader (reads modelcard + run.py)
│   │   ├── downloader.py       # Model pull/download
│   │   ├── optimizer.py        # ONNX/TensorRT/quantization
│   │   ├── profiler.py         # Hardware detection
│   │   ├── guardian.py         # Resource monitor + limits
│   │   ├── checkpoint.py       # Save/resume progress
│   │   └── output.py           # Standard output formatter
│   ├── cloud/
│   │   ├── router.py           # Smart backend selection
│   │   ├── colab_backend.py    # Google Colab backend
│   │   ├── kaggle_backend.py   # Kaggle backend
│   │   └── config.py           # Cloud credentials
│   ├── models/
│   │   ├── registry.py         # Load/manage model cards
│   │   └── cards/              # Model definitions (plugin format)
│   │       ├── yolov8/
│   │       │   ├── modelcard.yaml
│   │       │   └── run.py
│   │       ├── sam2/
│   │       │   ├── modelcard.yaml
│   │       │   └── run.py
│   │       ├── samurai/
│   │       │   ├── modelcard.yaml
│   │       │   └── run.py
│   │       ├── grounding_dino/
│   │       │   ├── modelcard.yaml
│   │       │   └── run.py
│   │       ├── depth_anything_v2/
│   │       │   ├── modelcard.yaml
│   │       │   └── run.py
│   │       └── florence2/
│   │           ├── modelcard.yaml
│   │           └── run.py
│   └── server/                 # Optional (pip install pixo[web])
│       ├── app.py
│       └── routes.py
└── tests/
```

---

## The Six Holes pixo Fills

| Problem | What everyone else does | What pixo does |
|---------|------------------------|----------------|
| Laptop freezes | Nothing — just crashes | Resource Guardian caps RAM/CPU, --background mode |
| Job crashes halfway | Start over from zero | Auto-checkpoint, resume from where it stopped |
| Dependency conflicts | "Figure it out yourself" | Isolated venv per model (v0.2) |
| No time estimate | Blank screen, no ETA | Pre-run estimate + live ETA based on actual speed |
| Different output formats | Every model dumps differently | Standard output structure, unified results.json |
| Setup complexity | README with 47 steps | `pixo pull model && pixo run model --input file` |

---

## Priority Order (v0.1 release path)

Build in this order — each phase delivers standalone value:

1. **Phase 5.1 — Plugin System** ← replace hardcoded runners, enable any model
2. **Phase 5.3 — Working Runners** ← 3 real models people can use
3. **Phase 6 — Checkpointing** ← killer feature, nobody else has this
4. **Phase 7.1 — Standard Output** ← clean results format
5. **v0.1.0 Release** ← ship to GitHub + PyPI
6. **Phase 5.2 — Isolated Envs** ← post-launch, adds robustness
7. **Phase 7.2 — Model Piping** ← post-launch, power user feature
8. **Phase 8 — Web Dashboard** ← post-launch, optional install
9. **Phase 9 — Community** ← growth, comes last
