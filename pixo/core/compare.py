"""Cross-model disagreement analysis for `pixo compare`.

Given detection results from two or more models on the same image, aligns
their boxes using IoU and classifies each detection as:
  - agreement: all models found a matching box with the same label
  - partial:   some models found it, others did not
  - unique:    only one model found it

v0.3 scope: supports Ultralytics detection models (yolov8, yolov11, yolov12, rtdetr)
on image inputs. Video compare and segmentation compare are planned for v0.4.
"""

SUPPORTED_MODELS = {"yolov8", "yolov11", "yolov12", "rtdetr"}

import base64
import html
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from pixo import __version__


@dataclass
class Detection:
    """A single box detection from one model."""
    bbox: tuple[float, float, float, float]  # x, y, w, h
    label: str
    score: float = 1.0
    model: str = ""


@dataclass
class MatchGroup:
    """A group of detections (across models) that refer to the same object."""
    label: str
    by_model: dict[str, Detection] = field(default_factory=dict)

    @property
    def models_present(self) -> list[str]:
        return sorted(self.by_model.keys())

    def representative_bbox(self) -> tuple[float, float, float, float]:
        """Average the boxes across models for display."""
        boxes = [d.bbox for d in self.by_model.values()]
        n = len(boxes)
        return tuple(sum(b[i] for b in boxes) / n for i in range(4))  # type: ignore


def detect_with_model(
    model_name: str,
    input_path: Path,
    conf: float = 0.25,
    device: str = "cpu",
) -> tuple[list[Detection], tuple[int, int]]:
    """Run a single detection model on an image and return (detections, (width, height))."""
    from pixo.core.plugin import loader
    from pixo.core.downloader import get_model_path, download_model

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"pixo compare v0.3 supports: {', '.join(sorted(SUPPORTED_MODELS))}. "
            f"Got '{model_name}'."
        )

    card = loader.load_card(model_name)
    variant = card.default_variant
    model_path = get_model_path(card.name, variant)
    if not model_path.exists():
        download_model(card)

    # Load via Ultralytics
    if model_name == "rtdetr":
        from ultralytics import RTDETR
        onnx = model_path.with_suffix(".onnx")
        model = RTDETR(str(onnx if onnx.exists() else model_path))
    else:
        from ultralytics import YOLO
        onnx = model_path.with_suffix(".onnx")
        model = YOLO(str(onnx if onnx.exists() else model_path), task="detect")

    preds = model.predict(
        source=str(input_path), device=device, save=False,
        verbose=False, conf=conf,
    )
    result = preds[0]

    # Image size (H, W, C) -> (W, H)
    h, w = result.orig_shape[:2]

    detections: list[Detection] = []
    if result.boxes is not None and len(result.boxes) > 0:
        xywh = result.boxes.xywh.cpu().numpy()  # center-x, center-y, w, h
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()
        for (cx, cy, bw, bh), sc, cls_idx in zip(xywh, confs, clses):
            # Convert center-xywh -> top-left xywh
            x = float(cx) - float(bw) / 2
            y = float(cy) - float(bh) / 2
            label = result.names[int(cls_idx)]
            detections.append(Detection(
                bbox=(x, y, float(bw), float(bh)),
                label=label,
                score=float(sc),
                model=model_name,
            ))

    return detections, (int(w), int(h))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU for boxes in (x, y, w, h) format."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def group_detections(
    detections_by_model: dict[str, list[Detection]],
    iou_threshold: float = 0.5,
) -> list[MatchGroup]:
    """Cluster detections across models into match groups using IoU + label.

    Uses a simple greedy algorithm: pick the highest-scoring unmatched detection
    from any model, then pull in matching detections (IoU >= threshold, same label)
    from every other model.
    """
    # Flatten and sort by score desc
    all_dets = []
    for m, dets in detections_by_model.items():
        for d in dets:
            d.model = m
            all_dets.append(d)
    all_dets.sort(key=lambda d: d.score, reverse=True)

    used: set[int] = set()
    groups: list[MatchGroup] = []

    for i, seed in enumerate(all_dets):
        if i in used:
            continue
        group = MatchGroup(label=seed.label)
        group.by_model[seed.model] = seed
        used.add(i)

        # Find best matching detection from each other model
        for j, cand in enumerate(all_dets):
            if j in used or cand.model in group.by_model:
                continue
            if cand.label != seed.label:
                continue
            if iou(seed.bbox, cand.bbox) >= iou_threshold:
                group.by_model[cand.model] = cand
                used.add(j)

        groups.append(group)

    return groups


def classify_groups(groups: list[MatchGroup], all_models: list[str]) -> dict[str, list[MatchGroup]]:
    """Split groups into agreements, partials, uniques."""
    n_models = len(all_models)
    agreements, partials, uniques = [], [], []
    for g in groups:
        present = len(g.by_model)
        if present == n_models:
            agreements.append(g)
        elif present == 1:
            uniques.append(g)
        else:
            partials.append(g)
    return {"agreements": agreements, "partials": partials, "uniques": uniques}


# --- HTML report ---

def _data_uri(path: Path) -> str:
    m, _ = mimetypes.guess_type(str(path))
    mime = m or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


_MODEL_COLORS = ["#0ea5e9", "#f97316", "#a855f7", "#10b981", "#ef4444", "#eab308"]


def build_compare_html(
    input_image: Path,
    models: list[str],
    groups: list[MatchGroup],
    image_size: tuple[int, int] | None = None,
) -> str:
    """Render a disagreement browser as a self-contained HTML string."""
    classified = classify_groups(groups, models)

    img_uri = _data_uri(input_image) if input_image.exists() else ""
    model_color = {m: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, m in enumerate(models)}

    def _box_svg(group: MatchGroup, scale: float = 1.0) -> str:
        svg_parts = []
        for m, det in group.by_model.items():
            x, y, w, h = det.bbox
            color = model_color[m]
            svg_parts.append(
                f'<rect x="{x * scale}" y="{y * scale}" width="{w * scale}" height="{h * scale}" '
                f'fill="none" stroke="{color}" stroke-width="2"/>'
            )
        return "".join(svg_parts)

    def _group_card(group: MatchGroup, kind: str) -> str:
        status_color = {
            "agreements": "#10b981",
            "partials": "#eab308",
            "uniques": "#ef4444",
        }[kind]
        present = ", ".join(
            f'<span style="color:{model_color[m]}">{html.escape(m)}</span>'
            for m in group.models_present
        )
        missing = [m for m in models if m not in group.by_model]
        missing_html = (
            f'<div class="missing">Missing: {", ".join(html.escape(m) for m in missing)}</div>'
            if missing else ""
        )

        x, y, w, h = group.representative_bbox()
        thumb_svg = (
            f'<svg viewBox="{x - 20} {y - 20} {w + 40} {h + 40}" width="120" height="120" '
            f'style="background:#0f172a; border-radius:6px">'
            f'{_box_svg(group)}'
            f'</svg>'
        )
        return (
            f'<div class="group" style="border-left:4px solid {status_color}">'
            f'  <div class="group-head">'
            f'    <span class="label">{html.escape(group.label)}</span>'
            f'    <span class="kind" style="background:{status_color}">{kind[:-1]}</span>'
            f'  </div>'
            f'  <div class="group-body">'
            f'    {thumb_svg}'
            f'    <div>'
            f'      <div class="present">Detected by: {present}</div>'
            f'      {missing_html}'
            f'    </div>'
            f'  </div>'
            f'</div>'
        )

    def _section(name: str, items: list[MatchGroup]) -> str:
        if not items:
            return f'<h2>{name.title()} ({len(items)})</h2><p class="empty">None.</p>'
        cards = "\n".join(_group_card(g, name) for g in items)
        return f'<h2>{name.title()} ({len(items)})</h2>{cards}'

    model_legend = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{model_color[m]}"></span>{html.escape(m)}</span>'
        for m in models
    )

    # Overlay SVG for the whole image showing every group
    iw, ih = image_size or (800, 600)
    overlay_boxes = []
    for g in groups:
        for m, det in g.by_model.items():
            x, y, w, h = det.bbox
            overlay_boxes.append(
                f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" '
                f'fill="none" stroke="{model_color[m]}" stroke-width="2"/>'
                f'<text x="{x + 2}" y="{y - 4}" fill="{model_color[m]}" font-size="12" '
                f'font-family="monospace">{html.escape(det.label)} ({html.escape(m)})</text></g>'
            )
    overlay_svg = (
        f'<svg viewBox="0 0 {iw} {ih}" preserveAspectRatio="xMidYMid meet" '
        f'style="position:absolute;inset:0;width:100%;height:100%">'
        f'{"".join(overlay_boxes)}'
        f'</svg>'
    )

    return _TEMPLATE.format(
        title=f"pixo compare — {html.escape(', '.join(models))}",
        models=", ".join(html.escape(m) for m in models),
        legend=model_legend,
        image_uri=img_uri,
        overlay_svg=overlay_svg,
        n_total=len(groups),
        n_agree=len(classified["agreements"]),
        n_partial=len(classified["partials"]),
        n_unique=len(classified["uniques"]),
        agreements_section=_section("agreements", classified["agreements"]),
        partials_section=_section("partials", classified["partials"]),
        uniques_section=_section("uniques", classified["uniques"]),
        pixo_v=html.escape(__version__),
    )


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 1100px; margin: 0 auto; padding: 32px 24px;
    background: #f8fafc; color: #0f172a;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0f172a; color: #e2e8f0; }}
    .card {{ background: #1e293b; border-color: #334155; }}
    .group {{ background: #1e293b; }}
  }}
  h1 {{ margin: 0 0 8px; font-size: 22px; }}
  h2 {{ margin-top: 28px; font-size: 16px; }}
  .sub {{ color: #64748b; font-size: 14px; margin-bottom: 16px; }}
  .stats {{ display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }}
  .stat {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; min-width: 120px; }}
  .stat .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat .value {{ font-size: 20px; font-weight: 600; }}
  .image-wrap {{ position: relative; display: inline-block; border-radius: 8px; overflow: hidden; }}
  .image-wrap img {{ max-width: 100%; display: block; }}
  .legend {{ display: flex; gap: 14px; margin: 8px 0 16px; font-size: 13px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
  .group {{
    background: white; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 10px;
  }}
  .group-head {{ display: flex; justify-content: space-between; align-items: center; }}
  .label {{ font-weight: 600; font-size: 14px; }}
  .kind {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    color: white; font-size: 10px; text-transform: uppercase; font-weight: 600;
  }}
  .group-body {{ display: flex; gap: 14px; margin-top: 10px; align-items: center; }}
  .present, .missing {{ font-size: 13px; }}
  .missing {{ color: #64748b; margin-top: 4px; }}
  .empty {{ color: #94a3b8; font-style: italic; }}
  footer {{ margin-top: 32px; font-size: 12px; color: #64748b; text-align: center; }}
</style>
</head>
<body>

<h1>pixo compare</h1>
<div class="sub">Models: <b>{models}</b></div>
<div class="legend">{legend}</div>

<div class="stats">
  <div class="stat"><div class="label">Total objects</div><div class="value">{n_total}</div></div>
  <div class="stat"><div class="label">All models agree</div><div class="value" style="color:#10b981">{n_agree}</div></div>
  <div class="stat"><div class="label">Partial agreement</div><div class="value" style="color:#eab308">{n_partial}</div></div>
  <div class="stat"><div class="label">Only one model</div><div class="value" style="color:#ef4444">{n_unique}</div></div>
</div>

<div class="image-wrap">
  <img src="{image_uri}"/>
  {overlay_svg}
</div>

{uniques_section}
{partials_section}
{agreements_section}

<footer>
  Generated with pixo {pixo_v}. Self-contained — no network calls to render this file.
</footer>
</body>
</html>
"""
