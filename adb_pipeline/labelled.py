"""Build ONE labelled image per frame that combines BOTH models:
  * Mapillary semantic segmentation (scene "stuff": barrier/grass/building/sky…)
    as a translucent colour overlay + legend, and
  * YOLO-World object detections as bounding boxes with labels + confidence.

Output: output/labelled/<sample_id>.jpg  (raw images stay in output/images/).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from . import config, stage6_lanes

LABELLED_DIR = config.LABELLED_DIR


def _font(sz):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def _box_color(label: str):
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    return (60 + h % 180, 60 + (h >> 8) % 180, 60 + (h >> 16) % 180)


def _scene_composition(seg) -> list[tuple[int, float, tuple, str]]:
    """Whole-image fraction of each SCENE_VIZ class present in `seg`, as
    (class_id, fraction, rgb, name) tuples. Same numbers the legend below
    renders and that get persisted per sample_id in
    config.SCENE_COMPOSITION_JSON.
    """
    total = seg.size
    present = []
    for cid, (rgb, name) in stage6_lanes.SCENE_VIZ.items():
        m = seg == cid
        if m.any():
            present.append((cid, m.sum() / total, rgb, name))
    return present


def _combined(img, seg, dets, feats, sid, direction, crash=None, meta=None):
    base = np.array(img).astype(np.float32)
    color = base.copy()
    present = _scene_composition(seg)
    for cid, _, rgb, _ in present:
        color[seg == cid] = rgb
    vis = (base * 0.6 + color * 0.4).astype(np.uint8)
    vis[seg == stage6_lanes.LANE_MARKING_GENERAL] = stage6_lanes.SCENE_VIZ[
        stage6_lanes.LANE_MARKING_GENERAL][0]
    vis = Image.fromarray(vis)
    d = ImageDraw.Draw(vis)
    big, small = _font(15), _font(12)

    # YOLO boxes on top
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        c = _box_color(det["label"])
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tag = f"{det['label'].split('/')[0]} {det['score']:.2f}"
        tw = d.textlength(tag, font=small)
        ty = max(0, y1 - 15)
        d.rectangle([x1, ty, x1 + tw + 4, ty + 15], fill=c)
        d.text((x1 + 2, ty + 1), tag, fill=(0, 0, 0), font=small)

    # --- info panel (top-left): segment facts from the GeoJSON ---
    m = meta or {}

    def _num(v, dec=0):
        try:
            f = float(v)
            return f"{f:.{dec}f}" if f == f else "n/a"   # f==f filters NaN
        except (TypeError, ValueError):
            return "n/a"

    def _txt(v):
        s = str(v).strip()
        return s if s and s.lower() != "nan" else "n/a"

    pol = m.get("pct_over")
    try:
        pol_s = f"{float(pol) * 100:.0f}%"
    except (TypeError, ValueError):
        pol_s = "n/a"
    over = direction == "over"
    lines = [
        (f"{sid}    {'OVER' if over else 'UNDER'} limit", (255, 235, 60)),
        (f"SpeedLimit {_num(m.get('sl'))}    F85thPercentileSpeed {_num(m.get('f85'))} km/h",
         (255, 255, 255)),
        (f"PercentOverLimit {pol_s}    RoadClass {_txt(m.get('road_class'))}", (170, 205, 255)),
        (f"LandUse {_txt(m.get('land_use'))}    est_lanes {feats['est_lane_count']}    "
         f"objects {len(dets)}", (170, 205, 255)),
        (f"Sinuosity {_num(m.get('sinuosity'), 2)}    "
         f"SharpCurves/km {_num(m.get('curves_per_km'), 2)}", (170, 205, 255)),
    ]
    pad, lh = 5, 16
    pw = max(d.textlength(t, font=small) for t, _ in lines) + 2 * pad
    ph = pad + lh * len(lines)
    d.rectangle([0, 0, pw, ph], fill=(0, 0, 0))
    yy = pad - 2
    for t, col in lines:
        d.text((pad, yy), t, fill=col, font=small)
        yy += lh

    # --- crash-site badge (top-right) ---
    n_cr = int(crash.get("n_crashes", 0)) if crash else 0
    if n_cr > 0:
        ns = int(crash.get("n_speeding", 0)); nc = int(crash.get("n_cutting", 0))
        badge = f"CRASH SITE: {n_cr} crash{'es' if n_cr != 1 else ''} ({ns} speeding, {nc} cutting-in)"
        bg = (200, 30, 30)
    else:
        badge = "NO recorded crash (<=120 m)"
        bg = (30, 130, 50)
    bw = d.textlength(badge, font=small)
    W = vis.size[0]
    d.rectangle([W - bw - 10, 0, W, 18], fill=bg)
    d.text((W - bw - 6, 2), badge, fill=(255, 255, 255), font=small)

    # --- scene legend (top-left, below the info panel) ---
    present_sorted = sorted(present, key=lambda p: p[1], reverse=True)
    y = ph + 4
    for _, cov, rgb, nm in [p for p in present_sorted if p[1] >= 0.01]:
        d.rectangle([5, y, 23, y + 13], fill=tuple(rgb), outline=(0, 0, 0))
        t = f"{nm} {cov*100:.0f}%"
        d.rectangle([26, y, 26 + d.textlength(t, font=small) + 4, y + 13], fill=(0, 0, 0))
        d.text((28, y), t, fill=(255, 255, 255), font=small)
        y += 15
    return vis


def run(detections: dict | None = None, samples: pd.DataFrame | None = None) -> int:
    if detections is None:
        detections = json.load(open(config.DETECTIONS_JSON, encoding="utf-8"))
    if samples is None:
        samples = pd.read_csv(config.SAMPLES_CSV)
    direction = dict(zip(samples["sample_id"], samples["mismatch_direction"]))

    # per-segment facts from the GeoJSON for the info panel.
    meta_cols = {"sl": "SpeedLimit", "f85": "F85thPercentileSpeed",
                 "pct_over": "PercentOverLimit", "road_class": "RoadClass",
                 "land_use": "LandUse"}
    meta_by_sid = {}
    have = {k: c for k, c in meta_cols.items() if c in samples.columns}
    mdf = samples.set_index("sample_id")
    for sid in mdf.index:
        meta_by_sid[sid] = {k: mdf.at[sid, c] for k, c in have.items()}

    # road curvature (geometry only) for the info panel — labelling, not analytics
    if "OBJECTID" in samples.columns:
        from . import curvature
        oid_by_sid = dict(zip(samples["sample_id"], samples["OBJECTID"]))
        cur = curvature.curvature_by_objectids(set(oid_by_sid.values()))
        for sid, meta in meta_by_sid.items():
            c = cur.get(int(oid_by_sid.get(sid, -1)), {})
            meta["sinuosity"] = c.get("sinuosity")
            meta["curves_per_km"] = c.get("curves_per_km")

    # crash lookup: sample_id -> crash counts, via the segment's OBJECTID.
    crash_by_sid = {}
    if config.SEGMENT_ACCIDENTS_CSV.exists() and "OBJECTID" in samples.columns:
        acc = pd.read_csv(config.SEGMENT_ACCIDENTS_CSV)
        cols = ["n_crashes", "n_speeding", "n_cutting"]
        merged = samples.merge(acc[["OBJECTID"] + cols], on="OBJECTID", how="left")
        for col in cols:
            merged[col] = merged[col].fillna(0)
        crash_by_sid = merged.set_index("sample_id")[cols].to_dict("index")
    else:
        print("[labelled] no segment_accidents.csv — run `accidents` for crash-site badges.")

    LABELLED_DIR.mkdir(parents=True, exist_ok=True)
    proc, model, dev = stage6_lanes._load_model()

    n = 0
    scene_by_sid: dict[str, dict[str, float]] = {}
    for sid, dets in tqdm(detections.items(), desc="labelled"):
        src = config.IMAGES_DIR / f"{sid}.jpg"
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        seg = stage6_lanes._segment(proc, model, dev, img)
        feats = stage6_lanes.lane_features(seg)
        vis = _combined(img, seg, dets, feats, sid, direction.get(sid, "under"),
                        crash=crash_by_sid.get(sid), meta=meta_by_sid.get(sid))
        vis.save(LABELLED_DIR / f"{sid}.jpg")
        scene_by_sid[sid] = {name: round(float(frac), 4)
                              for _, frac, _, name in _scene_composition(seg)}
        n += 1
    print(f"[labelled] wrote {n} combined images -> {LABELLED_DIR}")

    config.SCENE_COMPOSITION_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(scene_by_sid, open(config.SCENE_COMPOSITION_JSON, "w", encoding="utf-8"))
    print(f"[labelled] wrote {config.SCENE_COMPOSITION_JSON}")
    return n


if __name__ == "__main__":
    run()
