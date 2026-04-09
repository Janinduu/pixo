"""YOLOv11 runner -- real-time object detection using Ultralytics.

Same API as YOLOv8 -- Ultralytics handles all YOLO versions transparently.
"""

from pathlib import Path


def setup(model_dir: str, variant: str, device: str):
    """Load the YOLO11 model. Ultralytics auto-downloads if not found."""
    from ultralytics import YOLO

    model_path = Path(model_dir) / variant
    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists():
        return YOLO(str(onnx_path), task="detect")
    if model_path.exists():
        return YOLO(str(model_path), task="detect")
    return YOLO(variant, task="detect")


def run(model, input_path: str, output_dir: str, options: dict):
    """Run YOLO11 inference. Yields progress dicts."""
    import cv2

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = options.get("device", "cpu")
    conf = options.get("conf", 0.25)

    suffix = input_path.suffix.lower()
    is_video = suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    if not is_video:
        preds = model.predict(
            source=str(input_path), device=device, save=True,
            project=str(output_dir), name=".", exist_ok=True,
            verbose=False, conf=conf,
        )
        result = preds[0]
        classes = [result.names[int(c)] for c in result.boxes.cls]
        yield {"frame": 1, "total": 1, "objects": len(result.boxes), "classes": classes}
        return

    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    preds = model.predict(
        source=str(input_path), device=device, stream=True, save=True,
        project=str(output_dir), name=".", exist_ok=True,
        verbose=False, conf=conf,
    )

    total_objects = 0
    all_classes = set()

    for i, result in enumerate(preds, 1):
        total_objects += len(result.boxes)
        for cls in result.boxes.cls:
            all_classes.add(result.names[int(cls)])
        yield {
            "frame": i,
            "total": total_frames,
            "objects": total_objects,
            "classes": list(all_classes),
        }