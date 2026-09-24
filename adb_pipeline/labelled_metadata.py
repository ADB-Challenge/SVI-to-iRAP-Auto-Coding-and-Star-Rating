"""Build a labelled-image metadata CSV, mirroring the fetch metadata CSV
(config.MAPILLARY_METADATA_CSV) but for the final LABELLED output instead of
the raw fetch.

For each row that has a real fetched image (image_path not null -- i.e.
excluding ZERO_RESULTS/ERROR rows):
  - image_path is repointed from the raw fetch (config.IMAGES_DIR) to the
    combined labelled image (config.LABELLED_DIR)
  - detected_objects lists every YOLO detection for that image as
    "label:score" pairs, semicolon-separated (bounding boxes are left out --
    positional data not useful outside the image itself; still in
    detections.json if ever needed)
  - n_objects is the detection count, for quick sorting/filtering without
    parsing the detected_objects string
  - scene_composition lists Mask2Former's whole-image class fractions (road,
    vegetation/grass, building, etc.) as "name:pct%" pairs, semicolon-
    separated, >=1% only, largest first -- the exact same numbers and filter
    shown in the legend on the labelled image itself (see labelled.py)

Callers (e.g. SRS_pipeline/gather.py) must already have redirected
config.MAPILLARY_METADATA_CSV / DETECTIONS_JSON / SCENE_COMPOSITION_JSON /
LABELLED_DIR / LABELLED_METADATA_CSV to their own output folder before
calling run(), same convention as every other stage module in this pipeline.
"""

from __future__ import annotations

import json

import pandas as pd

from . import config


def _objects_str(dets: list[dict]) -> str:
    return ";".join(f"{d['label'].split('/')[0]}:{d['score']:.2f}" for d in dets)


def _scene_str(fracs: dict[str, float]) -> str:
    """Format as "name:pct%" pairs, >=1% only, largest first -- same filter
    and ordering as the legend rendered on the labelled image itself."""
    shown = sorted(((n, f) for n, f in fracs.items() if f >= 0.01),
                    key=lambda p: p[1], reverse=True)
    return ";".join(f"{name}:{frac*100:.0f}%" for name, frac in shown)


def run(metadata_csv=None) -> pd.DataFrame:
    """Build the labelled-image metadata CSV. Returns the resulting DataFrame."""
    if metadata_csv is None:
        metadata_csv = config.MAPILLARY_METADATA_CSV
    meta = pd.read_csv(metadata_csv)
    detections = json.load(open(config.DETECTIONS_JSON, encoding="utf-8"))
    scene = json.load(open(config.SCENE_COMPOSITION_JSON, encoding="utf-8"))

    have_image = meta[meta["image_path"].notna()].copy()
    have_image["image_path"] = have_image["sample_id"].apply(
        lambda sid: str(config.LABELLED_DIR / f"{sid}.jpg"))
    have_image["detected_objects"] = have_image["sample_id"].apply(
        lambda sid: _objects_str(detections.get(sid, [])))
    have_image["n_objects"] = have_image["sample_id"].apply(
        lambda sid: len(detections.get(sid, [])))
    have_image["scene_composition"] = have_image["sample_id"].apply(
        lambda sid: _scene_str(scene.get(sid, {})))

    config.LABELLED_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    have_image.to_csv(config.LABELLED_METADATA_CSV, index=False)
    print(f"[labelled-metadata] {len(have_image)}/{len(meta)} rows have a real image "
          "(rest omitted, e.g. ZERO_RESULTS)")
    print(f"[labelled-metadata] wrote {config.LABELLED_METADATA_CSV}")
    return have_image


if __name__ == "__main__":
    run()
