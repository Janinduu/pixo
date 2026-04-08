"""Checkpoint system — save and resume job progress."""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

CHECKPOINTS_DIR = Path.home() / ".pixo" / "checkpoints"


@dataclass
class JobState:
    job_id: str
    model: str
    variant: str
    input_path: str
    output_dir: str
    device: str
    status: str = "running"  # running, paused, completed, failed
    last_frame: int = 0
    total_frames: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    options: dict = field(default_factory=dict)

    @property
    def progress_percent(self) -> int:
        if self.total_frames == 0:
            return 0
        return int(self.last_frame / self.total_frames * 100)


class CheckpointManager:
    """Manages job checkpoints on disk."""

    def __init__(self, checkpoints_dir: Path = CHECKPOINTS_DIR):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, model: str, variant: str, input_path: str,
                   output_dir: str, device: str, options: dict | None = None) -> JobState:
        """Create a new job and return its state."""
        job_id = self._make_job_id(model, input_path)
        state = JobState(
            job_id=job_id,
            model=model,
            variant=variant,
            input_path=input_path,
            output_dir=output_dir,
            device=device,
            status="running",
            started_at=time.time(),
            updated_at=time.time(),
            options=options or {},
        )
        self.save_checkpoint(state)
        return state

    def save_checkpoint(self, state: JobState):
        """Save job state to disk."""
        state.updated_at = time.time()
        job_dir = self.checkpoints_dir / state.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        state_file = job_dir / "state.json"
        state_file.write_text(json.dumps(asdict(state), indent=2))

    def load_checkpoint(self, job_id: str) -> JobState | None:
        """Load a job state from disk."""
        state_file = self.checkpoints_dir / job_id / "state.json"
        if not state_file.exists():
            return None
        data = json.loads(state_file.read_text())
        return JobState(**data)

    def find_checkpoint(self, model: str, input_path: str) -> JobState | None:
        """Find an existing checkpoint for this model + input combo."""
        job_id = self._make_job_id(model, input_path)
        state = self.load_checkpoint(job_id)
        if state and state.status in ("paused", "running", "failed"):
            return state
        return None

    def mark_completed(self, state: JobState):
        """Mark a job as completed and keep the state (for history)."""
        state.status = "completed"
        self.save_checkpoint(state)

    def mark_paused(self, state: JobState):
        """Mark a job as paused."""
        state.status = "paused"
        self.save_checkpoint(state)

    def mark_failed(self, state: JobState):
        """Mark a job as failed."""
        state.status = "failed"
        self.save_checkpoint(state)

    def delete_checkpoint(self, job_id: str):
        """Remove a job's checkpoint directory."""
        job_dir = self.checkpoints_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)

    def list_jobs(self) -> list[JobState]:
        """List all jobs, newest first."""
        jobs = []
        if not self.checkpoints_dir.exists():
            return jobs
        for job_dir in self.checkpoints_dir.iterdir():
            if not job_dir.is_dir():
                continue
            state_file = job_dir / "state.json"
            if state_file.exists():
                try:
                    data = json.loads(state_file.read_text())
                    jobs.append(JobState(**data))
                except Exception:
                    continue
        jobs.sort(key=lambda j: j.updated_at, reverse=True)
        return jobs

    def get_latest_resumable(self) -> JobState | None:
        """Get the most recent paused or failed job."""
        for job in self.list_jobs():
            if job.status in ("paused", "failed"):
                return job
        return None

    def clean_completed(self) -> int:
        """Delete all completed job checkpoints. Returns count deleted."""
        count = 0
        for job in self.list_jobs():
            if job.status == "completed":
                self.delete_checkpoint(job.job_id)
                count += 1
        return count

    @staticmethod
    def _make_job_id(model: str, input_path: str) -> str:
        """Deterministic job ID from model + input path (so re-runs find existing checkpoints)."""
        key = f"{model}:{Path(input_path).resolve()}"
        return hashlib.sha256(key.encode()).hexdigest()[:10]
