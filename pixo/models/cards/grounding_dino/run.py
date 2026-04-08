"""Grounding DINO runner -- open-set object detection with text prompts."""

from pathlib import Path

VARIANT_REPOS = {
    "default": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}


def setup(model_dir: str, variant: str, device: str):
    """Load Grounding DINO zero-shot detection pipeline."""
    from transformers import pipeline

    repo = VARIANT_REPOS["default"]
    model_dir_name = Path(model_dir).name
    for key, repo_id in VARIANT_REPOS.items():
        if key in model_dir_name:
            repo = repo_id
            break

    dev = 0 if device == "cuda" else -1
    return pipeline("zero-shot-object-detection", model=repo, device=dev)


def run(model, input_path: str, output_dir: str, options: dict):
    """Run text-prompted detection. Yields progress dicts.

    Options:
        prompt (str): What to detect, e.g. "person" or "red car".
                      Multiple labels separated by commas: "person, car, dog"
        threshold (float): Confidence threshold (default 0.3)
    """
    import json
    import cv2
    import numpy as np
    from PIL import Image

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = options.get("prompt", "object")
    threshold = float(options.get("threshold", 0.3))
    labels = [l.strip() for l in prompt.split(",")]

    suffix = input_path.suffix.lower()
    is_video = suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    if is_video:
        yield from _run_video(model, input_path, output_dir, labels, threshold)
    else:
        yield from _run_image(model, input_path, output_dir, labels, threshold)


def _run_image(model, input_path, output_dir, labels, threshold):
    """Detect objects in a single image."""
    import json
    import cv2
    import numpy as np
    from PIL import Image

    image = Image.open(input_path).convert("RGB")
    yield {"frame": 0, "total": 1, "status": "detecting"}

    results = model(image, candidate_labels=labels, threshold=threshold)

    # Draw boxes on image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    all_labels = []
    detections = []

    for det in results:
        box = det["box"]
        label = det["label"]
        score = det["score"]
        all_labels.append(label)
        detections.append({
            "label": label,
            "score": round(score, 3),
            "box": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
        })

        cv2.rectangle(img_cv, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(img_cv, text, (box["xmin"], box["ymin"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(str(output_dir / f"{input_path.stem}_detected.jpg"), img_cv)

    # Save detections JSON
    meta = {"prompt": ", ".join(labels), "detections": detections, "count": len(detections)}
    (output_dir / "detections.json").write_text(json.dumps(meta, indent=2))

    yield {"frame": 1, "total": 1, "objects": len(detections), "classes": list(set(all_labels))}


def _run_video(model, input_path, output_dir, labels, threshold):
    """Detect objects in each frame of a video."""
    import json
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(input_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{input_path.stem}_detected.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    total_objects = 0
    all_classes = set()

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(pil_img, candidate_labels=labels, threshold=threshold)

        for det in results:
            box = det["box"]
            label = det["label"]
            score = det["score"]
            total_objects += 1
            all_classes.add(label)
            cv2.rectangle(frame, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (box["xmin"], box["ymin"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        yield {"frame": i + 1, "total": total, "objects": total_objects, "classes": list(all_classes)}

    cap.release()
    out.release()
