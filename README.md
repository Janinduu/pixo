<p align="center">
  <img src="logo.svg" alt="pixo" width="96" />
</p>

<p align="center">
  <strong>pixo</strong> — vision, simplified.
</p>

<p align="center">
  pixo is a runtime layer for computer vision models that lets you run, chain, and manage them as one system on your own machine.
</p>

<p align="center">
  <a href="https://pypi.org/project/pixo/"><img src="https://img.shields.io/pypi/v/pixo?color=7F77DD&style=flat-square" alt="PyPI" /></a>
  <a href="https://pypi.org/project/pixo/"><img src="https://img.shields.io/pypi/pyversions/pixo?color=7F77DD&style=flat-square" alt="Python" /></a>
  <a href="https://github.com/Janinduu/pixo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Janinduu/pixo?color=7F77DD&style=flat-square" alt="License" /></a>
</p>

```bash
pip install pixo
pixo try
```

That's it. `pixo try` auto-picks a model for your hardware, finds a sample image, runs it, and opens a browser report. You go from `pip install` to a working result in under a minute.

## Why pixo?

Running modern computer vision models locally is messy and fragile. pixo makes it safe, consistent, and easy to use multiple models without crashes, conflicts, or glue code.

| Problem | Without pixo | With pixo |
|---|---|---|
| Sensitive data leaks to cloud | Telemetry and version pings phone home | `--airgap` blocks every outbound call — provably local |
| First run takes 10 minutes | Read docs, pick model, find sample, type 4 commands | `pixo try` — one command, model picked for your hardware |
| Laptop freezes mid-inference | RAM/CPU maxed out, hard reboot | Resource Guardian caps usage automatically |
| Job crashes at 80% | Start over from zero | Auto-checkpoint + resume from the last save |
| Dependency conflicts | Packages break each other | Isolated environments per model |
| Inconsistent outputs | Every model outputs differently | Standard `results.json` + COCO / CSV exports |
| Sharing a result is painful | Zip + instructions + hope it opens | `pixo share` → single self-contained HTML file |

## Quick start

```bash
pip install pixo

# See it work (60-second hero demo)
pixo try

# Browse models and check your hardware
pixo list
pixo doctor

# Run any model on your own file
pixo run yolov8 --input photo.jpg
pixo run yolov8 --input video.mp4

# Speed up with ONNX (~40% faster on CPU)
pixo optimize yolov8
```

## Supported models

Every model carries a privacy badge — all nine bundled models run fully offline after weights are downloaded once.

| Model | Task | Default size |
|---|---|---|
| `yolov8` / `yolov11` / `yolov12` | Object detection | 5–7 MB |
| `rtdetr` | Transformer detection | 64 MB |
| `grounding_dino` | Text-prompted detection | 341 MB |
| `florence2` | Vision-language (caption, detect, OCR) | 460 MB |
| `depth_anything_v2` | Depth estimation | 99 MB |
| `sam2` | Segmentation | 898 MB |
| `samurai` | Video tracking + segmentation | 898 MB |

```bash
pixo run grounding_dino --input photo.jpg --prompt "red backpack, yellow hat"
pixo run florence2 --input photo.jpg --task caption
pixo run depth_anything_v2 --input photo.jpg
```

## Key features

### Local-first by design
```bash
pixo run yolov8 --input photo.jpg --airgap
```
Blocks all outbound network calls for the duration of the run — proves your data never left the machine. Combined with privacy badges on every model card, pixo is the only CV runtime built around "no bytes leave your laptop."

### See where models disagree
```bash
pixo compare yolov8 yolov11 yolov12 --input photo.jpg
```
Runs multiple detection models on the same image and shows only where they disagree. Agreement / partial / unique detections visualized side-by-side in a shareable HTML report.

### Shareable reports, no server
```bash
pixo share
```
Exports a run as a single self-contained `.html` file with images embedded. Attach it to a tweet, Slack, or email — anyone can open it in any browser.

### Safe on any laptop
```bash
pixo run yolov8 --input video.mp4 --low-memory --background
```
Resource Guardian caps RAM, CPU, and GPU usage. `--background` drops priority so your laptop stays responsive while pixo runs.

### Never lose progress
```bash
pixo run yolov8 --input long_video.mp4
# [Ctrl+C pauses, saves checkpoint]
pixo resume
```
Auto-saves every N frames. `Ctrl+C` pauses gracefully instead of killing the process.

### Free cloud GPUs
```bash
pixo setup-cloud --kaggle
pixo run sam2 --input photo.jpg --backend kaggle
```
Route heavy models to free Kaggle or Colab GPU. 30 hours/week free on Kaggle.

### Chain models
```bash
pixo pipe "grounding_dino -> sam2" --input photo.jpg --prompt "person"
```

### Browser UIs (optional)
```bash
pip install pixo[demo]
pixo serve yolov8          # Gradio UI for one model

pip install pixo[web]
pixo ui                    # Full local dashboard
```

## All commands

```bash
pixo try                   # Zero-setup hero demo
pixo list                  # List models with privacy badges
pixo info <model>          # Detailed model info
pixo pull <model>          # Pre-download a model
pixo run <model> -i <f>    # Run inference
pixo compare <m1> <m2> ... # Cross-model disagreement browser
pixo share [job_id]        # Export self-contained HTML report
pixo serve <model>         # Gradio browser UI
pixo ui                    # Full web dashboard
pixo pipe "m1 -> m2"       # Chain models
pixo doctor                # Check hardware
pixo optimize <model>      # ONNX conversion (~40% faster CPU)
pixo history               # Show past jobs
pixo resume [job_id]       # Resume a paused job
pixo view <job_id>         # Open a job's output folder
pixo setup-cloud           # Connect Kaggle / Colab
pixo cloud-status          # Cloud backend status
pixo rm <model>            # Remove a downloaded model
pixo upgrade               # Update pixo
pixo guide                 # In-terminal usage guide
```

## Python SDK

```python
import pixo

result = pixo.run("yolov8", input="photo.jpg")
print(result.objects, result.classes, result.time_seconds)

result = pixo.run("grounding_dino", input="photo.jpg", prompt="red car")
result = pixo.run("sam2", input="photo.jpg", backend="kaggle")

hw = pixo.doctor()
print(hw["ram_total_gb"], hw["has_gpu"])
```

## Adding a model

pixo uses a plugin system. Each model is two files:

```
pixo/models/cards/your_model/
  modelcard.yaml   # Metadata, dependencies, hardware, privacy
  run.py           # Two functions: setup() and run()
```

## Optional installs

pixo keeps the base install minimal. Pull only what you need:

```bash
pip install pixo[yolo]     # YOLO family (Ultralytics + OpenCV)
pip install pixo[onnx]     # ONNX Runtime for faster CPU
pip install pixo[vision]   # Grounding DINO / Florence-2 / SAM2
pip install pixo[cloud]    # Kaggle backend
pip install pixo[demo]     # Gradio for pixo serve
pip install pixo[web]      # FastAPI + uvicorn for pixo ui
pip install pixo[all]      # Everything
```

## Development

```bash
git clone https://github.com/Janinduu/pixo.git
cd pixo
pip install -e .
```

## Documentation

Detailed feature guides live in [`docs/`](docs/):

- [Features summary](docs/pixo_features_summary.docx) — one-page reference
- [Features explained](docs/pixo_features_detailed.docx) — full guide with why / what / how

## License

MIT
