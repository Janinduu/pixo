<p align="center">
  <img src="logo.svg" alt="pixo — vision, simplified." width="320" />
</p>

<p align="center">
  <strong>Run any computer vision model with one command</strong> — without freezing your laptop, losing progress, or fighting dependency hell.
</p>

<p align="center">
  <a href="https://pypi.org/project/pixo/"><img src="https://img.shields.io/pypi/v/pixo?color=7F77DD&style=flat-square" alt="PyPI" /></a>
  <a href="https://pypi.org/project/pixo/"><img src="https://img.shields.io/pypi/pyversions/pixo?color=7F77DD&style=flat-square" alt="Python" /></a>
  <a href="https://github.com/Janinduu/pixo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Janinduu/pixo?color=7F77DD&style=flat-square" alt="License" /></a>
</p>

```bash
pip install pixo
pixo pull yolov8
pixo run yolov8 --input photo.jpg
```

## Why pixo?

Running computer vision models (SAM2, YOLO, GroundingDINO, Depth Anything) on your own machine is painful. Your laptop freezes, jobs crash halfway, dependencies conflict, and there's no ETA. pixo fixes all of that.

| Problem | Without pixo | With pixo |
|---------|-------------|-----------|
| Laptop freezes | RAM/CPU maxed out, hard reboot | Resource Guardian caps usage, machine stays responsive |
| Job crashes at 80% | Start over from zero | Auto-checkpoint, resume from where it stopped |
| No time estimate | Stare at a blank screen | Pre-run estimate + live ETA |
| Setup complexity | README with 47 steps | `pixo pull model && pixo run model --input file` |

## Quick Start

```bash
# Install
pip install pixo

# See what's available
pixo list

# Download a model
pixo pull yolov8

# Run it
pixo run yolov8 --input photo.jpg
pixo run yolov8 --input video.mp4

# Check your hardware
pixo doctor

# Speed up with ONNX (40% faster on CPU)
pixo optimize yolov8
```

## Key Features

### Resource Guardian
Your laptop won't freeze. pixo checks resources before running and caps usage during execution.

```bash
# Safe for low-RAM machines (processes frame-by-frame)
pixo run yolov8 --input video.mp4 --low-memory

# Work normally while pixo runs in the background
pixo run yolov8 --input video.mp4 --background
```

### Free Cloud GPUs
Too slow locally? Route to free cloud GPUs automatically.

```bash
pixo setup-cloud          # Connect Kaggle/Colab (one-time)
pixo run yolov8 --input video.mp4 --backend kaggle  # Run on GPU for free
```

### Smart Routing
pixo estimates time per backend and picks the fastest option.

```bash
pixo run yolov8 --input video.mp4
# Local (CPU):  ~32 minutes
# Local (ONNX): ~19 minutes
# Kaggle (GPU): ~7 minutes  <-- recommended
```

## Supported Models

| Model | Task | Status |
|-------|------|--------|
| YOLOv8 | Object detection | Working |
| SAM2 | Image segmentation | Model card ready |
| GroundingDINO | Open-set detection | Model card ready |
| SAMURAI | Video tracking + segmentation | Model card ready |
| Depth Anything V2 | Depth estimation | Model card ready |
| Florence-2 | Vision-language | Model card ready |

## Commands

```bash
pixo pull <model>              # Download a model
pixo run <model> --input <file>  # Run inference
pixo list                      # List available models
pixo info <model>              # Show model details
pixo doctor                    # Check hardware
pixo optimize <model>          # Convert to ONNX
pixo setup-cloud               # Connect cloud accounts
pixo cloud-status              # Check cloud connections
pixo rm <model>                # Remove a downloaded model
```

## Adding a Model

pixo uses a plugin system. Each model is defined by two files:

```
pixo/models/cards/your_model/
  modelcard.yaml   # Model metadata, dependencies, hardware requirements
  run.py           # Two functions: setup() and run()
```

## Development

```bash
git clone https://github.com/Janinduu/pixo.git
cd pixo
pip install -e .
```

## License

MIT
