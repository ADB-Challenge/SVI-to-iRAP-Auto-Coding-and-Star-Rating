# SVI_to_iRAP

Street View imagery to iRAP Star Ratings for the ADB AI for Safer Roads
challenge (Thailand).

Selects high-traffic road segments from the ADB Innovation Thailand geojson,
fetches Mapillary street-level images every 500m along each segment, runs
YOLO-World object detection and Mask2Former scene segmentation on them,
encodes the results into the iRAP v3.10 (Drive on Left) upload schema, and
gets Star Ratings back from the ViDA Demonstrator.

## Setup

    uv sync
    cp .env.example .env

Fill in `.env`:

- `MAPILLARY_ACCESS_TOKEN` — from https://www.mapillary.com/dashboard/developers
- `VIDA_PHPSESSID` — ViDA Demonstrator session cookie (see
  [SRS_pipeline/README.md](SRS_pipeline/README.md#step-3--extract_irap_srspy));
  it expires, so refresh it after logging in again

`ADB_Innovation_Thailand.geojson` must be placed in this folder (it is not
committed). Model weights (YOLO-World, Mask2Former) download on first run.

torch and torchvision are pinned to the CUDA 12.8 wheel index (RTX 50-series).

## Run

From inside `SRS_pipeline/`, in order:

    uv run python gather.py              # Mapillary images + YOLO-World + Mask2Former (~1.5h)
    uv run python encode.py              # -> iRAP v3.10 upload CSV
    uv run python extract_irap_srs.py    # ViDA Star Ratings (~20 min)
    uv run python csvwkt2geojson.py      # -> GeoJSON for kepler.gl / QGIS

Before the last step, copy `output/main/enriched_irap_upload.csv` to
`output/main/SRS_rating(ViDA_API).csv` (the name `csvwkt2geojson.py` reads).

Outputs land in `output/main/`. Full details, flags and caveats are in
[SRS_pipeline/README.md](SRS_pipeline/README.md).

## Layout

- `SRS_pipeline/` — the four pipeline scripts
- `adb_pipeline/` — shared modules: config, Mapillary fetch, detection,
  segmentation, labelling, curvature, lanes
