"""pixo server — REST API + WebSocket for the web dashboard.

Start with: pixo ui
Or directly: uvicorn pixo.server.app:app --reload

Endpoints:
    GET  /api/models              List all models
    GET  /api/models/{name}       Model details
    POST /api/run                 Start a job
    GET  /api/jobs                List all jobs
    GET  /api/jobs/{id}           Job details
    GET  /api/hardware            Hardware profile
    GET  /api/cloud-status        Cloud backend status
    WS   /ws/progress/{job_id}    Live progress updates
"""

import asyncio
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="pixo",
    description="Run any computer vision model with one command",
    version="0.2.0",
)

# Allow CORS for the React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track running jobs and their progress
_jobs_progress: dict[str, dict] = {}


# --- Pydantic models ---

class RunRequest(BaseModel):
    model: str
    input_path: str
    output: str = "./pixo_output"
    device: str | None = None
    backend: str | None = None
    prompt: str | None = None
    task: str | None = None
    low_memory: bool = False


class RunResponse(BaseModel):
    job_id: str
    model: str
    status: str
    message: str


# --- API Routes ---

@app.get("/api/models")
def get_models():
    """List all available models, grouped by family."""
    import pixo
    models = pixo.list_models()

    # Build individual model list
    model_list = []
    for m in models:
        model_list.append({
            "name": m.name,
            "description": m.description,
            "task": m.task,
            "author": m.author,
            "variants": m.variants,
            "default_size_mb": m.default_size_mb,
            "downloaded": m.downloaded,
        })

    return model_list


# Model family grouping for the dashboard
MODEL_FAMILIES = {
    "yolo": {
        "display_name": "YOLO",
        "description": "Real-time object detection. Identifies and locates objects in images and video with bounding boxes.",
        "task": "detection",
        "versions": ["yolov8", "yolov11", "yolov12"],
    },
    "sam": {
        "display_name": "SAM",
        "description": "Segment Anything Model. Click or prompt to segment any object with pixel-perfect masks.",
        "task": "segmentation",
        "versions": ["sam2"],
    },
    "grounding_dino": {
        "display_name": "Grounding DINO",
        "description": "Text-prompted object detection. Describe what to find in natural language -- it locates it.",
        "task": "detection",
        "versions": ["grounding_dino"],
    },
    "depth": {
        "display_name": "Depth Anything",
        "description": "Monocular depth estimation. Generates depth maps from single images -- no stereo camera needed.",
        "task": "depth-estimation",
        "versions": ["depth_anything_v2"],
    },
    "florence": {
        "display_name": "Florence",
        "description": "Versatile vision-language model. Captioning, OCR, object detection, and more -- all in one model.",
        "task": "vision-language",
        "versions": ["florence2"],
    },
    "samurai": {
        "display_name": "SAMURAI",
        "description": "Video object tracking built on SAM2. Track any object across video frames automatically.",
        "task": "video-tracking-segmentation",
        "versions": ["samurai"],
    },
    "rtdetr": {
        "display_name": "RT-DETR",
        "description": "Real-Time Detection Transformer. End-to-end transformer detector -- no NMS needed, very accurate.",
        "task": "detection",
        "versions": ["rtdetr"],
    },
}


@app.get("/api/models/families")
def get_model_families():
    """Get models grouped by family for the dashboard."""
    import pixo
    all_models = {m.name: m for m in pixo.list_models()}

    families = []
    for family_key, family_info in MODEL_FAMILIES.items():
        versions = []
        for model_name in family_info["versions"]:
            if model_name in all_models:
                m = all_models[model_name]
                versions.append({
                    "name": m.name,
                    "variants": m.variants,
                    "default_size_mb": m.default_size_mb,
                    "downloaded": m.downloaded,
                })

        if versions:
            families.append({
                "family": family_key,
                "display_name": family_info["display_name"],
                "description": family_info["description"],
                "task": family_info["task"],
                "versions": versions,
            })

    return families


@app.get("/api/models/{name}")
def get_model(name: str):
    """Get detailed model info."""
    from pixo.core.plugin import loader
    try:
        card = loader.load_card(name)
    except KeyError:
        raise HTTPException(404, f"Model '{name}' not found")

    from pixo.core.downloader import is_downloaded
    variants = {}
    for vname, v in card.variants.items():
        variants[vname] = {
            "size_mb": v.size_mb,
            "description": v.description,
            "downloaded": is_downloaded(card.name, v, vname),
        }

    return {
        "name": card.name,
        "description": card.description,
        "task": card.task,
        "author": card.author,
        "version": card.version,
        "inputs": card.input_types,
        "outputs": card.output_types,
        "variants": variants,
        "hardware": {
            "min_ram_gb": card.hardware.min_ram_gb,
            "recommended_ram_gb": card.hardware.recommended_ram_gb,
            "min_vram_gb": card.hardware.min_vram_gb,
            "cpu_fallback": card.hardware.cpu_fallback,
        },
        "has_runner": loader.has_runner(name),
    }


@app.post("/api/run", response_model=RunResponse)
def start_run(req: RunRequest):
    """Start a model run (async — returns immediately, poll /api/jobs/{id} for status)."""
    import pixo

    # Validate input
    if not Path(req.input_path).exists():
        raise HTTPException(400, f"Input file not found: {req.input_path}")

    # Generate job ID
    job_id = str(int(time.time()))[-8:]
    _jobs_progress[job_id] = {"status": "starting", "model": req.model, "progress": 0}

    # Run in background thread
    def _run():
        try:
            _jobs_progress[job_id]["status"] = "running"
            result = pixo.run(
                req.model,
                input=req.input_path,
                output=req.output,
                device=req.device,
                backend=req.backend,
                prompt=req.prompt,
                task=req.task,
                low_memory=req.low_memory,
            )
            _jobs_progress[job_id] = {
                "status": "complete",
                "model": req.model,
                "progress": 100,
                "result": {
                    "objects": result.objects,
                    "classes": result.classes,
                    "time_seconds": result.time_seconds,
                    "output_dir": result.output_dir,
                },
            }
        except Exception as e:
            _jobs_progress[job_id] = {
                "status": "error",
                "model": req.model,
                "error": str(e),
            }

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return RunResponse(
        job_id=job_id,
        model=req.model,
        status="started",
        message=f"Job {job_id} started. Poll /api/jobs/{job_id} for status.",
    )


@app.get("/api/jobs")
def list_jobs():
    """List all jobs (from checkpoint manager + in-memory running jobs)."""
    from pixo.core.checkpoint import CheckpointManager
    mgr = CheckpointManager()

    jobs = []
    for job in mgr.list_jobs():
        jobs.append({
            "job_id": job.job_id[:8],
            "model": job.model,
            "input": job.input_path,
            "status": job.status,
            "progress": job.progress_percent,
            "date": str(getattr(job, "started_at", "")),
        })

    # Add in-memory running jobs
    for jid, info in _jobs_progress.items():
        if not any(j["job_id"] == jid for j in jobs):
            jobs.append({
                "job_id": jid,
                "model": info.get("model", ""),
                "input": "",
                "status": info.get("status", "unknown"),
                "progress": info.get("progress", 0),
                "date": "",
            })

    return jobs


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """Get job status and results."""
    if job_id in _jobs_progress:
        return _jobs_progress[job_id]

    from pixo.core.checkpoint import CheckpointManager
    mgr = CheckpointManager()
    for job in mgr.list_jobs():
        if job.job_id.startswith(job_id):
            return {
                "job_id": job.job_id[:8],
                "model": job.model,
                "status": job.status,
                "progress": job.progress_percent,
            }

    raise HTTPException(404, f"Job '{job_id}' not found")


@app.get("/api/hardware")
def get_hardware():
    """Get hardware profile."""
    import pixo
    return pixo.doctor()


@app.get("/api/cloud-status")
def get_cloud_status():
    """Get cloud backend status."""
    from pixo.cloud.config import load_config
    config = load_config()
    return {
        "kaggle": {
            "configured": config.kaggle.is_configured,
            "username": config.kaggle.username if config.kaggle.is_configured else None,
        },
        "colab": {
            "configured": config.colab.is_configured,
        },
    }


@app.post("/api/pull/{model_name}")
def pull_model(model_name: str):
    """Download a model."""
    import pixo
    try:
        pixo.pull(model_name)
        return {"status": "ok", "message": f"{model_name} downloaded"}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an input file for processing."""
    upload_dir = Path.home() / ".pixo" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"path": str(dest), "filename": file.filename, "size_mb": round(len(content) / 1024 / 1024, 2)}


# --- WebSocket for live progress ---

@app.websocket("/ws/progress/{job_id}")
async def progress_ws(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for live progress updates."""
    await websocket.accept()
    try:
        while True:
            if job_id in _jobs_progress:
                await websocket.send_json(_jobs_progress[job_id])
                if _jobs_progress[job_id].get("status") in ("complete", "error"):
                    break
            else:
                await websocket.send_json({"status": "unknown", "job_id": job_id})
            await asyncio.sleep(1)
    except Exception:
        pass
    finally:
        await websocket.close()


# --- Serve static files (React dashboard) ---

UI_DIR = Path(__file__).parent / "ui" / "dist"
if UI_DIR.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(str(UI_DIR / "index.html"))

    app.mount("/assets", StaticFiles(directory=str(UI_DIR / "assets")), name="assets")