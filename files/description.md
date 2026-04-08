# pixo — Ollama for Computer Vision

## What is pixo?

pixo is a free, open-source tool that lets you run **any** heavy computer vision
model on **any** laptop — without freezing your machine, losing progress,
or fighting dependency hell.

Think of it as **Ollama, but for vision models.**

With Ollama, you type `ollama run llama3` and a powerful LLM just works.
With pixo, you type `pixo run samurai --input video.mp4` and a powerful
CV model processes your video — no matter how weak your hardware is.

---

## The Problem

Running heavy CV models on normal laptops is painful in six specific ways:

### 1. Your laptop freezes
You run SAM2 or SAMURAI. Your entire machine locks up. Mouse won't move.
Can't switch tabs. Can't even force-quit. Only option: hard reboot.
This happens because the model eats all your RAM and CPU with no limits.

**Real numbers:**
| Setup | SAM2 on 5-min video |
|-------|---------------------|
| RTX 3060 laptop | ~7 minutes |
| i5 CPU-only laptop | ~2 to 3+ hours (if it doesn't freeze) |

### 2. Jobs crash and you lose everything
You're 80% through a 3-hour processing job. Colab disconnects. Or your
laptop overheats. Or power blinks. You start over from frame 1.

### 3. Dependency hell
Every model needs different packages. SAMURAI needs specific torch + sam2 versions.
Florence-2 needs different versions. Install one, break the other.
You spend hours debugging pip conflicts instead of doing actual work.

### 4. No one tells you if it'll work
You start a job with no idea if your hardware can handle it. No warning,
no estimate, no options. Just a blank screen and a prayer.

### 5. Every model outputs differently
SAM2 gives numpy masks. YOLO gives text files. Depth models give float arrays.
Want to chain them together? Write custom glue code every time.

### 6. Setup is a nightmare
Each model has a different README with 47 installation steps. Half the steps
are outdated. You're copying commands from GitHub issues hoping something works.

---

## How pixo Solves Each One

### 1. Resource Guardian — never freeze again
Before running, pixo checks your available RAM, CPU, and GPU. If the model
won't fit safely, it tells you and offers options:

```
$ pixo run samurai --input video.mp4

⚠ This model needs ~6GB RAM. You have 3.2GB free.
  This WILL freeze your laptop.

  Options:
  1. Close some apps and retry
  2. Use --low-memory mode (slower but safe)
  3. Route to Kaggle cloud GPU (~8 min)
```

During execution, pixo caps resource usage. Your laptop stays responsive.
Use `--background` mode to browse, code, or watch videos while pixo works.

### 2. Checkpointing — never lose progress
Every 100 frames, pixo saves your progress. If anything crashes:

```
$ pixo run samurai --input video.mp4

Found checkpoint: 8000/9582 frames (83%). Resume? [Y/n]
Resuming from frame 8001... done in 4 minutes.
```

Press Ctrl+C to pause. `pixo resume` to continue later.

### 3. Isolated environments — zero conflicts
Each model gets its own Python environment. SAMURAI and Florence-2
can coexist. `pixo pull` handles all dependencies automatically.

### 4. Smart estimation — know before you run
Before every job, pixo shows you what to expect:

```
Estimated times:
  Local (CPU, optimized): ~42 minutes
  Kaggle (T4 GPU):        ~8 minutes
  Recommended: Kaggle
```

### 5. Standard output — same format, every model
Every model outputs to the same structure: `results.json`, visualizations,
raw data, and COCO-format exports. Chain models with `pixo pipe`.

### 6. One command — just works
```bash
pip install pixo
pixo pull samurai
pixo run samurai --input video.mp4
```

That's it. No Docker. No CUDA debugging. No 47-step README.

---

## Who is this for?

- **Students** learning CV who can't afford GPUs and don't want to fight Colab
- **Researchers** who need to run experiments without wrecking their only laptop
- **Developers** prototyping CV apps who want to try models quickly
- **Anyone** who's ever had their laptop freeze running a heavy model

---

## Supported Model Categories

pixo isn't tied to specific models. It supports anything through its plugin system:

| Category | Example Models |
|----------|---------------|
| Object Detection | YOLOv8, GroundingDINO, OWLv2 |
| Segmentation | SAM2, Mask2Former, OneFormer |
| Video Tracking | SAMURAI, XMem, Cutie, Track Anything |
| Depth Estimation | Depth Anything V2, Metric3D, ZoeDepth |
| Vision-Language | Florence-2, Grounding SAM |
| Video Understanding | VideoMAE, InternVideo |

Adding a new model = one YAML file + a short Python script. Community-driven.

---

## How is this different from Ollama?

| | Ollama | pixo |
|---|---|---|
| Domain | LLMs (text) | CV models (images/video) |
| Input | Text prompt | Image, video, camera feed |
| Output | Text | Masks, boxes, labels, depth maps, tracks |
| Runs on | CPU (mostly) | GPU preferred, CPU with optimization |
| Cloud fallback | No (local only) | Yes (free: Kaggle, Colab) |
| Visualization | Terminal text | Visual overlays, frame scrubber |
| Resource protection | Not needed (LLMs are lighter) | Core feature (CV models freeze laptops) |
| Checkpointing | Not needed (responses are fast) | Core feature (video jobs take hours) |

The shared philosophy: **one command, it just works.**

---

## Tech Stack

- **Core CLI**: Python + Typer
- **Resource monitoring**: psutil + GPUtil
- **Model optimization**: ONNX Runtime, TensorRT (auto-detected)
- **Cloud backends**: Kaggle API (fully automated), Google Colab (semi-automated)
- **Environment isolation**: Python venvs (per-model)
- **Local server**: FastAPI
- **Web dashboard**: React + Vite + Tailwind
- **Model definitions**: YAML model cards + Python runner scripts
- **Package**: pip-installable (`pip install pixo`)
