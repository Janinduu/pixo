# pixo — Claude Code Prompts (v3)

Phases 1-4 are done. These prompts build the remaining features.
Run them in order. Each builds on the previous.

---

## Phase 5 — Plugin System

### Prompt 5.1 — Model Card Loader + Migration

```
Build the plugin system and migrate away from the old hardcoded config system.
This is a REPLACEMENT, not an addition — delete the old system entirely.

1. Create core/plugin.py with a PluginLoader class:
   - scan_models() → finds all model directories in pixo/models/cards/
   - load_model_card(name) → parses modelcard.yaml into a ModelCard dataclass
   - load_runner(name) → imports the model's run.py and returns setup() and run() functions
   - validate_card(path) → checks that a modelcard.yaml has all required fields

2. Define the ModelCard dataclass with these fields:
   name, description, version, task, author,
   source (type, repo, branch),
   dependencies (python version, packages list),
   hardware (min_ram_gb, recommended_ram_gb, min_vram_gb, recommended_vram_gb, cpu_fallback),
   inputs (list of type + formats),
   outputs (list of type + format),
   variants (dict of name → size_mb + description),
   checkpoint (supported boolean, every N frames)

3. Create model cards for these 6 models in pixo/models/cards/<name>/modelcard.yaml:
   - yolov8 (detection)
   - sam2 (segmentation)
   - samurai (video tracking + segmentation)
   - grounding_dino (open-set detection)
   - depth_anything_v2 (depth estimation)
   - florence2 (vision-language)

   For now, the run.py files can be stubs that print
   "Runner not yet implemented for {model}" — we'll fill them in later.
   But yolov8's run.py should work (port the existing runner code).

4. MIGRATION — remove the old system:
   - Delete pixo/models/configs/ directory (the old YAML configs)
   - Replace ModelConfig dataclass in registry.py with the new ModelCard format
     (or move registry logic into plugin.py and keep registry.py as a thin wrapper)
   - Delete the hardcoded RUNNERS dict from cli.py (line ~152)
   - Delete the _get_runner() function from cli.py
   - Update `pixo list`, `pixo info`, and `pixo run` to use PluginLoader

5. Make runner.py base class torch-free:
   - Remove `import torch` from the top of core/runner.py
   - Move get_device() to a utility function with lazy torch import
   - The base runner interface should work without torch installed
   - Each model's run.py imports torch (or whatever it needs) itself

6. Update the `pixo run` command flow:
   - Use PluginLoader to find the model's run.py
   - Call setup() to load the model
   - Call run() for inference
   - The plugin system is now the ONLY way models are loaded

The key principle: after this prompt, there is ONE config system (modelcard.yaml),
ONE way to load models (PluginLoader), and ZERO hardcoded model references in cli.py.
```

### Prompt 5.3 — Working Runners for Key Models

```
Implement actual working run.py files for the most important models.
These use the plugin format from 5.1 (setup + run functions).

1. **yolov8/run.py** — Port the existing YOLORunner to the plugin format:
   - setup(model_dir, variant, device): load the ultralytics model
   - run(model, input_path, output_dir, options): process image or video
   - Yield progress updates as dicts: {"frame": N, "total": total}
   - Support --conf and --classes options for filtering

2. **sam2/run.py** — SAM2 runner:
   - setup(): load SAM2 from facebook/sam2-hiera-large (or selected variant)
   - run() for images: automatic mask generation, save overlay + individual masks
   - run() for video: frame-by-frame segmentation, save mask sequence
   - Support --point and --box prompts via options dict
   - Yield progress: {"frame": N, "total": total_frames}

3. **depth_anything_v2/run.py** — Depth estimation runner:
   - setup(): load from LiheYoung/depth-anything-v2
   - run(): generate depth map, save as:
     a. Colored depth visualization (PNG)
     b. Raw depth values (numpy .npy file)
     c. Grayscale depth image
   - Support --max-resolution option to limit input size

For each runner:
- Keep it under 80 lines of code
- Handle both image and video inputs
- Yield progress dicts that the core plugin system understands
- Use try/except around the main processing loop
- Follow the exact setup()/run() interface from the plugin spec
- Import model-specific libraries (torch, ultralytics, etc.) inside
  the functions, not at module level — keeps pixo core lightweight

Test each one with: `pixo pull <model> && pixo run <model> --input test.jpg`
```

---

## Phase 6 — Checkpointing & Recovery

### Prompt 6.1 — Checkpoint System

```
Build the checkpointing system for pixo in core/checkpoint.py:

1. Create a CheckpointManager class:
   - create_job(model, input_path, options) → returns a job_id (short hash)
   - save_checkpoint(job_id, state) → saves progress to disk
     State includes:
     - frames_completed: list of frame indices done
     - last_frame: the last frame processed
     - model_state: any model state needed to resume (optional, model-specific)
     - partial_outputs: paths to outputs generated so far
     - started_at, updated_at timestamps
   - load_checkpoint(job_id) → loads saved state
   - has_checkpoint(model, input_path) → checks if a checkpoint exists for this job
   - delete_checkpoint(job_id) → clean up after job completes
   - list_jobs() → all jobs with their status

2. Storage format:
   ~/.pixo/checkpoints/
   └── abc123/
       ├── state.json        # progress metadata
       ├── partial_output/   # results generated so far
       └── model_state.pt    # model-specific state (optional)

3. Integrate with the core runner (via plugin system):
   - Before processing each frame/batch, check if checkpoint exists for this input
   - If yes → ask user: "Found checkpoint at 83%. Resume? [Y/n]"
   - During processing: save checkpoint every N frames (from modelcard.yaml)
   - After completion: delete checkpoint, keep only final results

4. Wire into CLI:
   - `pixo history` → shows all jobs (running, completed, paused, failed)
   - `pixo resume` → resumes the most recent paused/failed job
   - `pixo resume <job_id>` → resumes a specific job
   - `pixo jobs clean` → deletes old checkpoints to free disk space

5. Integrate with ResourceGuardian:
   - When guardian pauses due to high RAM/temperature → auto-save checkpoint
   - When guardian kills job due to critical temperature → save checkpoint first
   - Message: "Saved checkpoint at frame 8000. Run `pixo resume` when ready."
```

### Prompt 6.2 — Pause and Resume

```
Add manual pause/resume to pixo:

1. Signal handling in the core runner:
   - Register a signal handler for SIGINT (Ctrl+C)
   - First Ctrl+C: pause processing, save checkpoint, show:
     "Paused at frame 3200/9582. Run `pixo resume` to continue."
   - Second Ctrl+C within 3 seconds: actually kill the process
   - This means accidental Ctrl+C doesn't lose progress

2. Add keyboard hints during runs:
   - Press 'p' to pause (if terminal supports it)
   - Press 'q' to quit with checkpoint save
   - Press 's' to show current stats (RAM, CPU, temp, speed)
   - Show hint at start: "Press 'p' to pause, 'q' to save & quit"

3. Cloud checkpoint sync:
   - When running on Kaggle and session has < 5 minutes left:
     "Kaggle session ending in 5 minutes. Saving checkpoint..."
     Auto-save checkpoint to local disk (download from Kaggle)
   - `pixo resume` detects the cloud checkpoint and offers to:
     a. Resume on Kaggle (if available)
     b. Resume locally
     c. Resume on Colab

4. Update `pixo history` to show pause state:
   ID   Model     Input          Status     Progress   Date
   10   samurai   long_vid.mp4   paused     60%        yesterday
   
   `pixo resume 10` → picks up from 60%
```

---

## Phase 7.1 — Standard Output

### Prompt 7.1 — Output Formatter

```
Build the standard output system for pixo in core/output.py:

1. Create an OutputFormatter class:
   - create_output_dir(job_id, model, input_name) → creates structured output folder
   - save_results_json(job_id, metadata) → writes standard results.json
   - save_visualization(job_id, frames, overlays) → creates annotated images/video
   - save_raw(job_id, data_type, data) → saves masks/boxes/labels in standard format
   - export_coco(job_id) → converts results to COCO annotation format
   - export_csv(job_id) → converts results to simple CSV

2. Every `pixo run` output follows this structure:
   pixo_output/
   └── {job_id}_{model}_{input_name}/
       ├── results.json
       ├── summary.txt
       ├── visualizations/
       ├── raw/
       └── exports/

3. results.json schema:
   {
     "pixo_version": "0.1.0",
     "job_id": "abc123",
     "model": "samurai",
     "variant": "full",
     "input": {"path": "video.mp4", "frames": 9582, "resolution": "1920x1080"},
     "device": "cpu",
     "mode": "background",
     "resource_usage": {"peak_ram_gb": 4.2, "avg_cpu_percent": 65},
     "timing": {"total_seconds": 2520, "fps": 3.8},
     "results_summary": {"objects_detected": 24, "tracks": 8},
     "output_files": {...}
   }

4. Update ALL model runners (run.py files) to use OutputFormatter
   instead of custom output logic. The runner produces raw data —
   OutputFormatter handles all file organization.

5. Add CLI commands:
   - `pixo view <job_id>` — opens the visualization folder in system file manager
   - `pixo view <job_id> --web` — starts a simple local HTML viewer
```

---

## ===== v0.1.0 RELEASE PREP =====

### Prompt — Release Preparation

```
Prepare pixo for v0.1.0 public release:

1. Update README.md with:
   - Clear installation instructions
   - Quickstart (pull → run → view results)
   - List of supported models with descriptions
   - Key features (Resource Guardian, checkpointing, cloud routing)
   - GIF/screenshot placeholders for demo
   - Contributing guide (how to add a model via modelcard.yaml + run.py)
   - Link to plan/roadmap for upcoming features

2. Add LICENSE file (MIT)

3. Add `pixo upgrade` command to cli.py:
   - Runs `pip install --upgrade pixo` via subprocess
   - Shows current version vs latest version
   - Simple, one-liner implementation

4. Clean up pyproject.toml:
   - Verify all dependencies are correct and minimal
   - Add optional [web] dependency group for FastAPI + uvicorn
   - Add project URLs (homepage, repository, issues)
   - Add classifiers and keywords for PyPI discovery

5. Verify all commands work end-to-end:
   - pixo list
   - pixo pull yolov8
   - pixo run yolov8 --input test.jpg
   - pixo run yolov8 --input test.jpg --low-memory
   - pixo run yolov8 --input test.jpg --background
   - pixo doctor
   - pixo history
   - pixo resume
   - pixo view <job_id>
```

---

## Post-Launch Prompts

### Prompt 5.2 — Isolated Environments (v0.2)

```
Build per-model virtual environment isolation for pixo:

1. Create core/envmanager.py:
   - create_env(model_name) → creates a venv at ~/.pixo/envs/{model_name}/
   - install_deps(model_name) → reads dependencies from modelcard.yaml,
     pip installs them into the model's venv
   - get_python(model_name) → returns path to the model's venv python binary
   - env_exists(model_name) → checks if env is already set up
   - delete_env(model_name) → removes environment and all installed packages

2. Integrate with `pixo pull`:
   When user runs `pixo pull samurai`:
   a. Download model weights (existing logic)
   b. Create venv: ~/.pixo/envs/samurai/
   c. Install dependencies from modelcard.yaml into that venv
   d. Show progress: "Installing dependencies for samurai... [████████] done"
   e. Test import: try importing the main package to verify it works

3. Integrate with `pixo run`:
   When user runs `pixo run samurai --input video.mp4`:
   - Execute the model's run.py using the model's own venv python
   - Use subprocess with the venv's python binary
   - Pass input/output paths and options via command line args or JSON
   - Stream stdout back for progress updates

4. Handle edge cases:
   - If venv creation fails → clear error message, suggest manual fix
   - If dependency install fails → show which package failed
   - If two models share large packages (like torch) → each has its own copy.
     Disk space is cheap, dependency conflicts are expensive.
   - Add `pixo env list` to show all environments and their sizes
   - Add `pixo env clean <model>` to rebuild an environment from scratch

5. Default behavior:
   - By default, models install into user's current environment (simple path)
   - `--isolate` flag or config option enables per-model venvs
   - `--no-env` flag for advanced users who manage their own environment

The venv approach prevents "I installed SAMURAI and now my YOLO environment
is broken" — but it's opt-in for v0.2 to avoid complexity at launch.
```

### Prompt 7.2 — Model Piping (v0.2)

```
Build the model piping system for pixo:

1. Create core/pipeline.py:
   - parse_pipeline("grounding_dino → sam2 → samurai") → list of model names
   - run_pipeline(models, input_path, options) → executes models in sequence
   - Between each step:
     a. Take output from previous model
     b. Convert to input format needed by next model
     c. Pass along with relevant metadata (bounding boxes, masks, etc.)

2. Define standard intermediate formats:
   - Detection output → list of bounding boxes + labels (JSON)
   - Segmentation output → masks (PNG) + metadata (JSON)
   - Tracking output → tracked masks + IDs across frames (JSON)
   These are the "connectors" between models.

3. Create converter functions:
   - detection_to_segmentation_input() → convert boxes to SAM2 prompts
   - segmentation_to_tracking_input() → convert masks to tracker initialization

4. Wire into CLI:
   pixo pipe "grounding_dino → sam2" --input video.mp4 --prompt "person"
   
   Shows step-by-step progress:
   Step 1/2: grounding_dino — detecting "person"... done (found 4 objects)
   Step 2/2: sam2 — segmenting 4 objects... [████████] 100%

5. Pre-built pipelines as YAML files:
   - "detect_and_segment": grounding_dino → sam2
   - "detect_and_track": grounding_dino → sam2 → samurai
   - "segment_and_depth": sam2 → depth_anything_v2
   
   User can run: `pixo pipe detect_and_track --input video.mp4 --prompt "car"`
```

### Prompt 8.1 — FastAPI Backend (v0.3)

```
Build the FastAPI server for pixo's web dashboard in server/app.py.
This is an optional install: `pip install pixo[web]`

Endpoints:

GET  /api/models              → list all models with status
GET  /api/models/{name}       → model details (from modelcard)
POST /api/run                 → start a job (model, input file, options)
GET  /api/jobs                → list all jobs (running, done, paused, failed)
GET  /api/jobs/{id}           → job details + progress
GET  /api/jobs/{id}/results   → download results
POST /api/jobs/{id}/pause     → pause a running job
POST /api/jobs/{id}/resume    → resume a paused job
GET  /api/hardware            → hardware profile from pixo doctor
GET  /api/cloud-status        → cloud backend availability
WS   /ws/progress/{job_id}    → real-time progress + resource usage

Reuse all existing CLI logic — the API is a thin wrapper around the
same core functions. Don't duplicate business logic.

Add `pixo ui` command that starts the server and opens browser.
Serve static files from ui/dist/ for the frontend.
```

### Prompt 8.2 — React Dashboard (v0.3)

```
Build the React frontend for pixo's web dashboard.

Tech: React 18 + Vite + Tailwind CSS
Design: Clean, minimal, dark theme. Developer-friendly.

Pages:

1. **Models page (home)**
   - Grid of model cards grouped by task
   - Each card: name, description, task badge, size, "Pull" or "Run" button
   - Downloaded models have a green indicator
   - Search bar at top

2. **Run page**
   - Pick model, upload input (drag and drop)
   - Options panel: variant, device, low-memory toggle, background toggle
   - Resource pre-check display
   - Live progress: progress bar, resource graph, ETA

3. **Results page**
   - Image: original | annotated side by side
   - Video: frame scrubber with overlay toggle
   - Download buttons: full results, masks, video, COCO JSON

4. **Jobs page**
   - Table of all jobs with pause/resume buttons
   - Click job → view results

5. **Settings page**
   - Cloud connections, resource limit sliders, default preferences

Connect to FastAPI backend. WebSocket for live progress.
```

---

## Utility Prompts

### Add any new model to pixo

```
Add support for [MODEL_NAME] to pixo:

1. Research the model:
   - Find the official repo and HuggingFace weights
   - Check what dependencies it needs
   - Determine input types (image/video) and output types

2. Create pixo/models/cards/[model_name]/modelcard.yaml:
   - Fill in all fields following the modelcard spec
   - List exact dependencies with version constraints
   - Set realistic hardware requirements

3. Create pixo/models/cards/[model_name]/run.py:
   - Implement setup(model_dir, variant, device) → loads model
   - Implement run(model, input_path, output_dir, options) → runs inference
   - Yield progress updates as dicts
   - Keep it under 80 lines
   - Handle both image and video if the model supports it
   - Import model-specific libraries inside functions, not at module level

4. Test the full flow:
   pixo pull [model_name]
   pixo run [model_name] --input test_image.jpg
   pixo run [model_name] --input test_video.mp4 --low-memory
```

### Debug / fix prompt

```
I'm getting this error when running pixo:

[PASTE ERROR]

The relevant file is [FILE PATH].

Fix the issue. Explain what went wrong in one sentence.
Don't add comments to code unless the fix is non-obvious.
```

### Benchmark prompt

```
Run a benchmark of pixo on my machine:

1. Pull yolov8 (should already be downloaded)
2. Run it on a test image in three modes:
   - Normal: pixo run yolov8 --input test.jpg
   - ONNX optimized: pixo run yolov8 --input test.jpg (with optimized model)
   - Low-memory: pixo run yolov8 --input test.jpg --low-memory
3. Record time and peak RAM for each
4. Print a comparison table

Then do the same for a short video clip (first 100 frames).
Save results to benchmark_results.json.
```

---

## Order of Execution

```
v0.1.0 Release Path:
  5.1 (plugin system + migration)
  5.3 (working runners)
  6.1 (checkpoint system)
  6.2 (pause/resume)
  7.1 (standard output)
  Release prep
  --- SHIP v0.1.0 ---

Post-Launch (v0.2+):
  5.2 (isolated environments)
  7.2 (model piping)
  8.1 (FastAPI backend)
  8.2 (React dashboard)
```

Each prompt number is one Claude Code session.
Test after each. Don't skip ahead.
