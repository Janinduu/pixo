"""Standard output formatter — unified output structure for all models."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pixo import __version__


@dataclass
class JobResults:
    """Standard results metadata for every pixo run."""
    pixo_version: str = __version__
    job_id: str = ""
    model: str = ""
    variant: str = ""
    input: dict = field(default_factory=dict)
    device: str = ""
    mode: str = "normal"
    timing: dict = field(default_factory=dict)
    results_summary: dict = field(default_factory=dict)
    output_files: dict = field(default_factory=dict)


class OutputFormatter:
    """Creates and manages structured output directories."""

    def __init__(self, base_dir: str, model: str, input_name: str, job_id: str = ""):
        self.model = model
        self.input_name = Path(input_name).stem
        self.job_id = job_id[:8] if job_id else str(int(time.time()))[-8:]

        # Create structured output directory
        dir_name = f"{self.job_id}_{model}_{self.input_name}"
        self.root = Path(base_dir) / dir_name
        self.viz_dir = self.root / "visualizations"
        self.raw_dir = self.root / "raw"
        self.exports_dir = self.root / "exports"

        for d in (self.root, self.viz_dir, self.raw_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._results = JobResults(job_id=self.job_id, model=model)

    @property
    def results(self) -> JobResults:
        return self._results

    def set_input_info(self, path: str, frames: int = 0, resolution: str = ""):
        """Set input file metadata."""
        self._results.input = {
            "path": str(path),
            "frames": frames,
            "resolution": resolution,
        }

    def set_timing(self, total_seconds: float, fps: float = 0.0):
        """Set timing information."""
        self._results.timing = {
            "total_seconds": round(total_seconds, 1),
            "fps": round(fps, 2) if fps else 0,
        }

    def set_summary(self, **kwargs):
        """Set results summary (model-specific)."""
        self._results.results_summary = kwargs

    def save_results_json(self) -> Path:
        """Write results.json to the output directory."""
        # Collect output files
        files = {}
        for subdir in (self.viz_dir, self.raw_dir, self.exports_dir):
            rel_name = subdir.name
            found = [str(f.relative_to(self.root)) for f in subdir.rglob("*") if f.is_file()]
            if found:
                files[rel_name] = found
        self._results.output_files = files

        path = self.root / "results.json"
        path.write_text(json.dumps(asdict(self._results), indent=2))
        return path

    def save_summary_txt(self, extra_lines: list[str] | None = None) -> Path:
        """Write a human-readable summary.txt."""
        r = self._results
        lines = [
            f"pixo {r.pixo_version} — {r.model}",
            f"Input: {r.input.get('path', '?')}",
            f"Device: {r.device}",
        ]
        if r.timing:
            secs = r.timing.get("total_seconds", 0)
            mins, s = divmod(int(secs), 60)
            lines.append(f"Time: {mins}m {s}s")
            if r.timing.get("fps"):
                lines.append(f"Speed: {r.timing['fps']} fps")
        if r.results_summary:
            lines.append("")
            for k, v in r.results_summary.items():
                lines.append(f"{k}: {v}")
        if extra_lines:
            lines.extend([""] + extra_lines)

        path = self.root / "summary.txt"
        path.write_text("\n".join(lines))
        return path

    def export_coco(self, detections: list[dict], image_width: int, image_height: int) -> Path:
        """Export detection results in COCO annotation format.

        Each detection should have: {"bbox": [x,y,w,h], "label": str, "score": float, "frame": int}
        """
        categories = {}
        cat_id = 1
        annotations = []
        ann_id = 1
        images = {}

        for det in detections:
            # Build category map
            label = det.get("label", "object")
            if label not in categories:
                categories[label] = cat_id
                cat_id += 1

            # Build image entries
            frame = det.get("frame", 0)
            if frame not in images:
                images[frame] = {
                    "id": frame,
                    "file_name": f"frame_{frame:06d}.jpg",
                    "width": image_width,
                    "height": image_height,
                }

            annotations.append({
                "id": ann_id,
                "image_id": frame,
                "category_id": categories[label],
                "bbox": det.get("bbox", [0, 0, 0, 0]),
                "score": det.get("score", 1.0),
                "area": det.get("bbox", [0, 0, 0, 0])[2] * det.get("bbox", [0, 0, 0, 0])[3],
                "iscrowd": 0,
            })
            ann_id += 1

        coco = {
            "images": list(images.values()),
            "annotations": annotations,
            "categories": [
                {"id": cid, "name": name} for name, cid in categories.items()
            ],
        }

        path = self.exports_dir / "coco.json"
        path.write_text(json.dumps(coco, indent=2))
        return path

    def export_csv(self, rows: list[dict]) -> Path:
        """Export results as CSV.

        Each row should be a flat dict with the same keys.
        """
        if not rows:
            return self.exports_dir / "results.csv"

        import csv
        path = self.exports_dir / "results.csv"
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return path
