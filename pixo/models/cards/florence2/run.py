"""Florence-2 runner -- vision-language model for captioning, detection, OCR.

NOTE: Florence-2's custom HuggingFace code requires transformers<5.0.
With transformers>=5.0, use --isolate with a pinned environment, or
install: pip install transformers==4.49.0
"""

from pathlib import Path

VARIANT_REPOS = {
    "default": "microsoft/Florence-2-base",
    "large": "microsoft/Florence-2-large",
}

TASKS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "detect": "<OD>",
    "ocr": "<OCR>",
}


def setup(model_dir: str, variant: str, device: str):
    """Load Florence-2 model and processor."""
    import transformers

    repo = VARIANT_REPOS["default"]
    model_dir_name = Path(model_dir).name
    for key, repo_id in VARIANT_REPOS.items():
        if key in model_dir_name:
            repo = repo_id
            break

    # Check transformers version — Florence-2 custom code is incompatible with v5+
    major_ver = int(transformers.__version__.split(".")[0])
    if major_ver >= 5:
        raise RuntimeError(
            f"Florence-2 requires transformers<5.0 (you have {transformers.__version__}).\n"
            f"Fix: pip install transformers==4.49.0\n"
            f"Or use --isolate to run in an isolated environment with pinned deps."
        )

    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)

    if device == "cuda":
        import torch
        model = model.to("cuda")

    return {"model": model, "processor": processor, "device": device}


def run(model_dict, input_path: str, output_dir: str, options: dict):
    """Run Florence-2 inference. Yields progress dicts.

    Options:
        task (str): caption, detailed_caption, detect, or ocr (default: caption)
        prompt (str): Optional text prompt for grounded tasks
    """
    import json
    import cv2
    import numpy as np
    from PIL import Image

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model_dict["model"]
    processor = model_dict["processor"]
    device = model_dict["device"]

    task_name = options.get("task", "caption")
    task_prompt = TASKS.get(task_name, "<CAPTION>")
    text_input = options.get("prompt", "")
    if text_input:
        prompt = task_prompt + text_input
    else:
        prompt = task_prompt

    image = Image.open(input_path).convert("RGB")
    yield {"frame": 0, "total": 1, "status": f"running {task_name}"}

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    generated = model.generate(**inputs, max_new_tokens=1024, num_beams=3)
    decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        decoded, task=task_prompt, image_size=(image.width, image.height)
    )

    # Save result
    output_data = {}
    classes = []

    if task_name in ("caption", "detailed_caption"):
        caption = result.get(task_prompt, "")
        output_data = {"task": task_name, "caption": caption}
        classes = ["caption"]

    elif task_name == "detect":
        det_result = result.get(task_prompt, {})
        bboxes = det_result.get("bboxes", [])
        labels = det_result.get("labels", [])
        classes = list(set(labels))

        # Draw boxes on image
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / f"{input_path.stem}_detected.jpg"), img_cv)

        output_data = {"task": "detect", "count": len(bboxes),
                       "labels": labels, "bboxes": bboxes}

    elif task_name == "ocr":
        ocr_text = result.get(task_prompt, "")
        output_data = {"task": "ocr", "text": ocr_text}
        classes = ["text"]

    (output_dir / "florence2_result.json").write_text(json.dumps(output_data, indent=2))

    obj_count = len(output_data.get("bboxes", [])) or 1
    yield {"frame": 1, "total": 1, "objects": obj_count, "classes": classes or [task_name]}
