"""Stage 6 — estimate the number of lanes from each Street View image.

Uses Mask2Former (Mapillary Vistas semantic) to segment the road and lane
markings, then derives lane features by counting longitudinal lane-marking lines
crossing a scan band near the ego vehicle.

IMPORTANT: this is an *approximate* estimate. Google Street View is forward
facing but not a dashcam; faded markings, occlusion, intersections and the 90°
crop all add noise. Use the continuous features (marking fraction, line count)
as much as the integer lane estimate.

Outputs: output/lanes.csv (per image) and optional segmentation overlays in
output/labeled/lanes_viz/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import config

MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
# Mapillary Vistas class ids (from the model config)
LANE_MARKING_GENERAL = 24
ROAD = 13
SERVICE_LANE = 14
BIKE_LANE = 7
CROSSWALK_PLAIN = 8
DRIVABLE_IDS = [ROAD, SERVICE_LANE, BIKE_LANE, CROSSWALK_PLAIN]

# "Stuff" classes YOLO-World/LVIS cannot detect — captured here as the fraction
# of the image each covers. These directly answer "barriers, grass, buildings".
SCENE_CLASSES = {
    "barrier_frac": 5, "guardrail_frac": 4, "wall_frac": 6, "fence_frac": 3,
    "curb_frac": 2, "sidewalk_frac": 15, "building_frac": 17, "vegetation_frac": 30,
    "terrain_frac": 29, "pole_frac": 45, "utility_pole_frac": 47,
    # elevated mass-transit / overpass structures (MRT/skytrain viaducts)
    "bridge_frac": 16,
}

LANES_CSV = config.LANES_CSV
VIZ_DIR = config.PIPELINE_DIR / "scene_viz"

# Color + name for each visualized Mapillary class (BGR-agnostic RGB).
SCENE_VIZ = {
    13: ((90, 90, 90), "road"),
    14: ((110, 110, 140), "service lane"),
    24: ((255, 235, 0), "lane marking"),
    23: ((255, 190, 0), "crosswalk marking"),
    5: ((255, 0, 0), "BARRIER"),
    4: ((255, 130, 0), "guard rail"),
    6: ((255, 0, 200), "wall"),
    3: ((255, 110, 180), "fence"),
    2: ((0, 210, 210), "curb"),
    15: ((0, 200, 255), "sidewalk"),
    17: ((180, 110, 60), "building"),
    30: ((0, 175, 0), "vegetation/grass"),
    29: ((140, 160, 50), "terrain"),
    16: ((140, 0, 220), "BRIDGE/elevated transit"),
    45: ((245, 245, 245), "pole"),
    47: ((205, 205, 205), "utility pole"),
}


def _load_model():
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    dev = "cuda" if (config.MODEL_NN_DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID).to(dev).eval()
    print(f"[stage6] {MODEL_ID} on {dev}")
    return proc, model, dev


def lane_features(seg: np.ndarray) -> dict:
    """Derive lane features from a (H,W) semantic-segmentation map.

    Counts longitudinal lane-marking lines by scanning many rows across the road's
    depth (not just the bottom, where perspective fans lines off-frame). For each
    row we count distinct marking runs; the robust line count is the largest run
    count seen in several rows. Lanes = lines - 1.
    """
    from collections import Counter

    H, W = seg.shape
    total = H * W
    lane = seg == LANE_MARKING_GENERAL
    road = np.isin(seg, DRIVABLE_IDS)
    road_px = int(road.sum())

    gap = max(4, int(0.015 * W))          # min horizontal gap between two lines
    y0, y1 = int(0.50 * H), int(0.93 * H)  # scan the road from mid-image to near ego
    per_row = []
    for y in range(y0, y1, 2):
        cols = np.where(lane[y])[0]
        if cols.size == 0:
            continue
        runs = 1 + int((np.diff(cols) > gap).sum())
        per_row.append(runs)

    if per_row:
        freq = Counter(per_row)
        # largest line-count seen in >= 3 scan rows (ignores single-row noise)
        robust = [v for v, f in freq.items() if f >= 3]
        lines = max(robust) if robust else int(np.median(per_row))
        lines = min(lines, 8)
    else:
        lines = 0

    if road_px < 0.02 * total:
        est = 0  # essentially no visible road
    elif lines >= 2:
        est = lines - 1  # N boundary lines -> N-1 lanes
    else:
        est = 1  # road visible but 0-1 markings -> assume single lane visible

    feats = {
        "road_frac": round(road_px / total, 4),
        "lane_marking_frac": round(int(lane.sum()) / max(road_px, 1), 4),
        "num_marking_lines": int(lines),
        "est_lane_count": int(est),
    }
    # scene composition: fraction of the image covered by each "stuff" class
    for name, cid in SCENE_CLASSES.items():
        feats[name] = round(float((seg == cid).sum()) / total, 4)
    # convenience flag the user asked about explicitly
    feats["has_barrier"] = int(feats["barrier_frac"] + feats["guardrail_frac"]
                               + feats["wall_frac"] > 0.01)
    return feats


def _segment(proc, model, dev, img):
    import torch

    inp = proc(images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inp)
    seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.size[::-1]])[0]
    return seg.cpu().numpy()


def _overlay(img: Image.Image, seg: np.ndarray, feats: dict, sid: str) -> Image.Image:
    """Paint every recognised scene class over the image, with a legend.

    Each Mapillary "stuff" class (barrier, wall, vegetation/grass, building, sky,
    sidewalk, lanes...) is colour-coded and alpha-blended, so what the model sees
    is directly visible. Lane markings are drawn solid on top.
    """
    from PIL import ImageDraw, ImageFont

    base = np.array(img).astype(np.float32)
    color = base.copy()
    present = []  # (coverage, color, name) for the legend
    total = seg.size
    for cid, (rgb, name) in SCENE_VIZ.items():
        mask = seg == cid
        cov = mask.sum()
        if cov == 0:
            continue
        if cid == LANE_MARKING_GENERAL:
            color[mask] = rgb  # solid for thin markings
        else:
            color[mask] = rgb
        present.append((cov / total, rgb, name))

    alpha = 0.5
    vis = (base * (1 - alpha) + color * alpha).astype(np.uint8)
    # keep lane markings crisp/solid
    vis[seg == LANE_MARKING_GENERAL] = SCENE_VIZ[LANE_MARKING_GENERAL][0]
    vis = Image.fromarray(vis)

    d = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 15)
        small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 13)
    except Exception:
        font = small = ImageFont.load_default()

    cap = f"{sid}   est lanes={feats['est_lane_count']}   marking lines={feats['num_marking_lines']}"
    d.rectangle([0, 0, d.textlength(cap, font=font) + 10, 22], fill=(0, 0, 0))
    d.text((5, 3), cap, fill=(255, 255, 0), font=font)

    # legend (classes covering >= 0.5% of the frame), largest first
    present.sort(reverse=True)
    rows = [(c, rgb, nm) for c, rgb, nm in present if c >= 0.005]
    y = 28
    for cov, rgb, nm in rows:
        d.rectangle([5, y, 25, y + 14], fill=tuple(rgb), outline=(0, 0, 0))
        label = f"{nm}  {cov*100:.0f}%"
        d.rectangle([28, y, 28 + d.textlength(label, font=small) + 4, y + 14], fill=(0, 0, 0))
        d.text((30, y + 1), label, fill=(255, 255, 255), font=small)
        y += 17
    return vis


def run(samples: pd.DataFrame | None = None, limit: int | None = None,
        viz: bool = False) -> pd.DataFrame:
    config.ensure_dirs()
    if samples is None:
        samples = pd.read_csv(config.SAMPLES_CSV)
    if viz:
        VIZ_DIR.mkdir(parents=True, exist_ok=True)

    ids = samples["sample_id"].tolist()
    if limit:
        ids = ids[:limit]
    proc, model, dev = _load_model()

    rows = []
    for sid in tqdm(ids, desc="lanes"):
        src = config.IMAGES_DIR / f"{sid}.jpg"
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        seg = _segment(proc, model, dev, img)
        feats = lane_features(seg)
        feats["sample_id"] = sid
        rows.append(feats)
        if viz:
            _overlay(img, seg, feats, sid).save(VIZ_DIR / f"{sid}.jpg")

    lead = ["sample_id", "est_lane_count", "num_marking_lines", "lane_marking_frac",
            "road_frac", "has_barrier"]
    df = pd.DataFrame(rows)
    df = df[lead + [c for c in df.columns if c not in lead]]
    if not limit:
        df.to_csv(LANES_CSV, index=False)
        print(f"[stage6] wrote {LANES_CSV}")
    dist = df["est_lane_count"].value_counts().sort_index().to_dict()
    print(f"[stage6] processed {len(df)} images | est_lane_count distribution: {dist}")
    return df


if __name__ == "__main__":
    run()
