"""Stage 2 (alternative) — fetch a Mapillary street-level image per sampled
segment, instead of Google Street View (see stage2_gsv.py).

For each sample we already computed a representative (lat, lon) and a road
heading in Stage 1. Mapillary has no camera control (no heading/fov/pitch like
Google's Static Street View) — it only serves images contributors have already
captured. So we:
  1. Query the Images endpoint (graph.mapillary.com) for a small bbox around the
     point to list nearby existing captures.
  2. Rank candidates by distance to the target point, then by how close their
     ``compass_angle`` is to the target heading, and pick the best one.
  3. Download that image's pre-rendered thumbnail URL.

When ``MAPILLARY_ACCESS_TOKEN`` is absent (config.MAPILLARY_MOCK), we generate a
deterministic synthetic placeholder image instead, so Stages 3-4 remain fully
runnable.

Some Mapillary captures are 360-degree spherical panoramas (equirectangular
projection) rather than normal photos -- a different geometry than YOLO-World
/ Mask2Former expect (straight lines curve, the same object can appear twice).
We request ``camera_type``/``is_pano`` from the API and, when a pick is
spherical, reproject a normal-looking forward-facing crop out of it aimed at
the sample's target heading -- roughly recreating what a perspective camera
would have captured. Fisheye captures are flagged but not corrected (fixing
lens distortion needs per-lens ``camera_parameters`` we don't attempt here).

Also logs Mapillary's own ``quality_score`` (0-1, API-provided) per image, as
a signal for spotting likely-corrupted captures without needing our own
pixel-level heuristics.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

from . import config

_IMAGES_URL = "https://graph.mapillary.com/images"
_M_PER_DEG_LAT = 111_320.0


def _mock_image(path: Path, lat: float, lon: float, heading: float) -> dict:
    """Deterministic placeholder so the pipeline runs without a real API token."""
    seed = int(hashlib.md5(f"{lat},{lon}".encode()).hexdigest(), 16)
    w, h = (int(x) for x in config.MAPILLARY_MOCK_IMAGE_SIZE.split("x"))
    img = Image.new("RGB", (w, h), (seed % 90 + 100, (seed >> 8) % 90 + 100, 150))
    d = ImageDraw.Draw(img)
    # crude "road" so detectors have some structure to chew on
    d.polygon([(w * 0.45, h), (w * 0.55, h), (w * 0.52, h * 0.55), (w * 0.48, h * 0.55)],
              fill=(60, 60, 60))
    d.text((10, 10), f"MOCK Mapillary\n{lat:.5f},{lon:.5f}\nhead {heading:.0f}",
           fill=(255, 255, 255))
    img.save(path)
    return {"status": "MOCK", "image_id": None, "captured_at": None,
            "pano_lat": lat, "pano_lon": lon, "compass_angle": None, "distance_m": 0.0,
            "camera_type": None, "is_pano": False, "reprojected": False,
            "quality_score": None}


def _bbox_around(lat: float, lon: float, radius_m: float) -> str:
    coslat = math.cos(math.radians(lat))
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / (_M_PER_DEG_LAT * max(coslat, 1e-6))
    return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"


def _search_images(lat: float, lon: float, attempts: int = 3) -> list[dict]:
    """Query the Images endpoint, retrying empty results a few times.

    The bbox search is flaky for sparse coverage -- the exact same query can
    alternate between empty and non-empty a few seconds apart (see
    SRS_pipeline/gather.py's _mapillary_has_image for how this was
    confirmed by hand).
    """
    last_exc = None
    for i in range(attempts):
        try:
            r = requests.get(
                _IMAGES_URL,
                params={
                    "access_token": config.MAPILLARY_ACCESS_TOKEN,
                    "fields": ("id,captured_at,compass_angle,geometry,camera_type,is_pano,"
                               f"quality_score,{config.MAPILLARY_IMAGE_FIELD}"),
                    "bbox": _bbox_around(lat, lon, config.MAPILLARY_SEARCH_RADIUS_M),
                    "limit": config.MAPILLARY_MAX_RESULTS,
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            if data:
                return data
        except Exception as e:
            last_exc = e
        if i < attempts - 1:
            time.sleep(1.0)
    if last_exc is not None:
        raise last_exc
    return []


def _equirect_to_perspective(img: Image.Image, yaw_deg: float, fov_deg: float = 90.0,
                              out_w: int = 640, out_h: int = 640) -> Image.Image:
    """Extract a normal-looking forward-facing crop from a 360 equirectangular
    panorama, aimed `yaw_deg` clockwise from the panorama's own front direction
    (its horizontal center column). Nearest-neighbour sampling -- good enough
    for downstream detection/segmentation, not photographic quality.
    """
    src = np.asarray(img.convert("RGB"))
    H, W = src.shape[:2]

    fov = math.radians(fov_deg)
    f = (out_w / 2) / math.tan(fov / 2)

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x = (u - out_w / 2 + 0.5) / f
    y = -(v - out_h / 2 + 0.5) / f
    z = np.ones_like(x, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm

    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y

    theta = np.arctan2(x, z)              # -pi..pi around the vertical axis
    phi = np.arcsin(np.clip(y, -1.0, 1.0))  # -pi/2..pi/2

    src_x = np.clip(((theta / (2 * math.pi) + 0.5) * W).astype(np.int32), 0, W - 1)
    src_y = np.clip(((0.5 - phi / math.pi) * H).astype(np.int32), 0, H - 1)

    return Image.fromarray(src[src_y, src_x])


def _fix_if_spherical(path: Path, image: dict, heading: float) -> bool:
    """If `image` is a 360 spherical capture, overwrite the file at `path` with
    a reprojected forward-facing crop aimed at `heading`. Returns True if a fix
    was applied."""
    if image.get("camera_type") != "spherical" and not image.get("is_pano"):
        return False
    compass = image.get("compass_angle")
    yaw = 0.0 if compass is None else (heading - compass) % 360
    pano = Image.open(path)
    fixed = _equirect_to_perspective(pano, yaw)
    fixed.save(path)
    return True


def _pick_best(candidates: list[dict], lat: float, lon: float, heading: float) -> dict | None:
    """Closest existing capture by distance, tie-broken by heading match."""
    if not candidates:
        return None
    coslat = math.cos(math.radians(lat))

    def score(c: dict) -> tuple[float, float]:
        clon, clat = c["geometry"]["coordinates"]
        dist = math.hypot((clat - lat) * _M_PER_DEG_LAT, (clon - lon) * _M_PER_DEG_LAT * coslat)
        angle = c.get("compass_angle")
        heading_diff = 180.0 if angle is None else abs((angle - heading + 180) % 360 - 180)
        return (dist, heading_diff)

    best = min(candidates, key=score)
    best["_distance_m"], best["_heading_diff"] = score(best)
    return best


def _fetch_image(path: Path, image: dict) -> None:
    url = image.get(config.MAPILLARY_IMAGE_FIELD)
    if not url:
        raise ValueError(f"no {config.MAPILLARY_IMAGE_FIELD} on image {image.get('id')}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path.write_bytes(r.content)


def run(samples: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fetch images for all samples. Returns a fetch-log DataFrame.

    Drop-in replacement for stage2_gsv.run() — same input/output shape, so
    downstream stages (stage3_detect, labelled, stage6_lanes) don't need to
    change no matter which stage2 module produced the DataFrame.
    """
    config.ensure_dirs()
    if samples is None:
        samples = pd.read_csv(config.SAMPLES_CSV)

    mode = "MOCK (no access token)" if config.MAPILLARY_MOCK else "LIVE Mapillary"
    print(f"[stage2] fetching {len(samples)} images — mode: {mode}")

    records = []
    for _, r in tqdm(samples.iterrows(), total=len(samples), desc="Mapillary"):
        sid = r["sample_id"]
        lat, lon, heading = float(r["lat"]), float(r["lon"]), float(r["heading"])
        img_path = config.IMAGES_DIR / f"{sid}.jpg"
        rec = {"sample_id": sid, "lat": lat, "lon": lon, "heading": heading,
               "image_path": str(img_path)}

        try:
            if config.MAPILLARY_MOCK:
                rec.update(_mock_image(img_path, lat, lon, heading))
            else:
                candidates = _search_images(lat, lon)
                best = _pick_best(candidates, lat, lon, heading)
                if best is not None:
                    clon, clat = best["geometry"]["coordinates"]
                    rec["status"] = "OK"
                    rec["image_id"] = best.get("id")
                    rec["captured_at"] = best.get("captured_at")
                    rec["pano_lat"], rec["pano_lon"] = clat, clon
                    rec["compass_angle"] = best.get("compass_angle")
                    rec["distance_m"] = round(best["_distance_m"], 1)
                    rec["camera_type"] = best.get("camera_type")
                    rec["is_pano"] = bool(best.get("is_pano"))
                    rec["quality_score"] = best.get("quality_score")
                    _fetch_image(img_path, best)
                    rec["reprojected"] = _fix_if_spherical(img_path, best, heading)
                else:
                    rec["status"] = "ZERO_RESULTS"  # no nearby Mapillary coverage
                    rec["image_path"] = None
        except Exception as e:  # network/HTTP errors shouldn't kill the batch
            rec["status"] = f"ERROR: {e}"
            rec["image_path"] = None
        records.append(rec)

    log = pd.DataFrame(records)
    log.to_csv(config.MAPILLARY_METADATA_CSV, index=False)
    ok = (log["image_path"].notna()).sum()
    print(f"[stage2] images saved: {ok}/{len(log)} -> {config.IMAGES_DIR}")
    print(f"[stage2] wrote {config.MAPILLARY_METADATA_CSV}")
    return log


if __name__ == "__main__":
    run()
