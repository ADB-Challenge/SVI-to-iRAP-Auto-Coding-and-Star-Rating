"""Stage 3 — promptless open-vocabulary object detection with YOLO-World.

Unlike a prompt-conditioned detector, we do **not** hand a curated class list.
YOLO-World is loaded with its full vocabulary (LVIS = 1,203 everyday-object
categories by default) so it reports whatever it recognises in each image. Every
detected object is kept verbatim and carried into the correlation stage.

Runs on GPU (config.YOLO_DEVICE). Output: ``detections.json`` mapping each
sample_id to a list of {label, score, box[xyxy]} detections.
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd
from tqdm import tqdm

from . import config

# Whole-word matcher for on-road vehicles, so "car" doesn't catch "cart"/"railcar".
_VEHICLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in config.VEHICLE_KEYWORDS) + r")\b"
)


def _is_vehicle(label: str) -> bool:
    return bool(_VEHICLE_RE.search(label.lower()))


def _is_excluded(label: str) -> bool:
    """Drop on-road vehicles and specific noise classes (weathervane, can)."""
    if config.EXCLUDE_VEHICLES and _is_vehicle(label):
        return True
    return label.split("/")[0].strip().lower() in config.OBJECT_EXCLUDE


def _vocabulary() -> list[str]:
    """Return the class names the detector is allowed to recognise."""
    if config.YOLO_VOCAB == "coco":
        return []  # empty => keep the model's built-in COCO vocabulary
    # LVIS 1,203 classes ship with ultralytics as a dataset config.
    import ultralytics
    import yaml

    p = os.path.join(os.path.dirname(ultralytics.__file__),
                     "cfg", "datasets", "lvis.yaml")
    names = yaml.safe_load(open(p, encoding="utf-8"))["names"]
    vocab = [names[i] for i in sorted(names)]
    # add extra prompts (MRT/overpass structures) not present in LVIS
    vocab += [c for c in config.YOLO_EXTRA_CLASSES if c not in vocab]
    return vocab


def _load_model():
    import torch
    from ultralytics import YOLO

    device = config.YOLO_DEVICE
    if device != "cpu" and not torch.cuda.is_available():
        print("[stage3] CUDA not available — falling back to CPU.")
        device = "cpu"

    model = YOLO(config.YOLO_MODEL)
    vocab = _vocabulary()
    if vocab:
        model.set_classes(vocab)  # promptless: the whole LVIS taxonomy
        print(f"[stage3] {config.YOLO_MODEL} | vocab={config.YOLO_VOCAB} "
              f"({len(vocab)} classes) | device={device}")
    else:
        print(f"[stage3] {config.YOLO_MODEL} | vocab=coco "
              f"({len(model.names)} classes) | device={device}")
    return model, device


def run(fetch_log: pd.DataFrame | None = None) -> dict:
    """Detect objects across all fetched images. Returns {sample_id: [dets]}."""
    config.ensure_dirs()
    if fetch_log is None:
        fetch_log = pd.read_csv(config.GSV_METADATA_CSV)

    have_img = fetch_log[fetch_log["image_path"].notna()].reset_index(drop=True)
    print(f"[stage3] images to process: {len(have_img)}")
    if have_img.empty:
        print("[stage3] nothing to detect.")
        return {}

    model, device = _load_model()
    names = model.names  # index -> label (reflects set_classes vocabulary)

    detections: dict[str, list[dict]] = {}
    rows = have_img.to_dict("records")
    bs = max(1, config.YOLO_BATCH)
    for start in tqdm(range(0, len(rows), bs), desc="detect"):
        chunk = rows[start : start + bs]
        paths = [r["image_path"] for r in chunk]
        try:
            results = model.predict(
                paths,
                conf=config.YOLO_CONF,
                iou=config.YOLO_IOU,
                max_det=config.YOLO_MAX_DET,
                device=device,
                verbose=False,
            )
        except Exception as e:
            print(f"[stage3] batch at {start} failed: {e}")
            for r in chunk:
                detections[r["sample_id"]] = []
            continue

        for r, res in zip(chunk, results):
            dets = []
            for box in res.boxes:
                cls_idx = int(box.cls.item())
                label = names[cls_idx]
                if _is_excluded(label):
                    continue  # drop vehicles + noise classes (weathervane, can)
                dets.append(
                    {
                        "label": label,
                        "score": round(float(box.conf.item()), 4),
                        "box": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                    }
                )
            detections[r["sample_id"]] = dets

    with open(config.DETECTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)
    total = sum(len(v) for v in detections.values())
    distinct = len({d["label"] for v in detections.values() for d in v})
    print(f"[stage3] {total} objects ({distinct} distinct labels) across "
          f"{len(detections)} images -> {config.DETECTIONS_JSON}")
    return detections


if __name__ == "__main__":
    run()
