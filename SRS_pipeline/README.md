# SRS pipeline

Selects high-traffic-flow road segments from the ADB Innovation Thailand
geojson, samples Street View imagery every 500m along each segment, runs
object detection + scene segmentation on the imagery, encodes the results
into the iRAP v3.10 upload schema, and submits them to ViDA for Star Ratings.

## Prerequisites

Run once from the project root (`clean_adb/`, one level up from this folder):

    uv sync
    cp .env.example .env      # then fill in MAPILLARY_ACCESS_TOKEN and VIDA_PHPSESSID

`ADB_Innovation_Thailand.geojson` must be present at the project root
(`config.GEOJSON_PATH` points there by default). The first real run also
needs internet access once, to auto-download the Mask2Former model
(`facebook/mask2former-swin-large-mapillary-vistas-semantic`) from Hugging
Face — it isn't bundled in this repo.

All commands below are run from inside `SRS_pipeline/`.

## Step 1 — `gather.py`

    uv run python gather.py

- Selects every ADB segment with `AnalysisStatus == "Valid"` and
  `Percentile` between 0.75 and 1.0 (the top quartile by traffic flow),
  capped at `MAX_SEGMENTS` (50) segments per run.
- Walks each segment's **full drawn geometry**, taking a reading every
  500m (`POINT_INTERVAL_M`) from start to end.
- Checks Mapillary coverage at each segment's midpoint first; segments with
  no nearby imagery are skipped entirely before the expensive per-reading
  fetch starts.
- Fetches a Mapillary image at every kept reading, runs YOLO-World object
  detection + Mask2Former scene segmentation on each image, and computes
  per-segment curvature (`curves_per_km`) from the geojson's own geometry.
- Writes `output/main/road_attributes_raw.csv` — one row per reading, with
  the raw CV/curvature signal alongside the ADB attributes.

Useful flags:

    uv run python gather.py --sample-only   # stop after sampling, see the point
                                             # count before committing to a full
                                             # Mapillary+YOLO+Mask2Former run
    uv run python gather.py --use-cache     # if samples.csv and labelled_metadata.csv
                                             # already exist from a prior run, reuse
                                             # them instead of re-calling the Mapillary
                                             # API / re-running detection — just rebuilds
                                             # road_attributes_raw.csv from what's on disk

## Step 2 — `encode.py`

    uv run python encode.py

Reads `output/main/road_attributes_raw.csv` and encodes each reading into
the full 72-column iRAP v3.10 upload schema (one row per reading, matching
genuine iRAP survey practice of short discrete sections rather than one row
per multi-km segment).

A handful of columns (Roadworks, Median type, Roadside severity object) use
CV-based inference rules layered on top of the raw YOLO/Mask2Former output —
see [RULES.md](RULES.md) for the rationale, confidence level, and empirical
fire-rate behind each rule before trusting its output blindly. Everything
else not directly observable from the ADB geojson or the CV pipeline falls
back to a documented default (see the constants at the top of `encode.py`).

Writes:
- `output/main/irap_upload.csv` — the 72-column ViDA upload file
- `output/main/rule_fire_log.csv` — per-reading trace of which CV inference
  rules fired, for auditing

## Step 3 — `extract_irap_srs.py`

    uv run python extract_irap_srs.py

Submits `output/main/irap_upload.csv` to the ViDA Demonstrator API
(model v3.10, Drive on Left) and pulls back Star Ratings (car, motorcycle,
pedestrian, bicycle) for every row.

Before running: open the ViDA Demonstrator in a browser, log in, and copy a
fresh `PHPSESSID` cookie value (DevTools → Application → Cookies →
`demonstrator.vida.irap.org`) into `VIDA_PHPSESSID` in `.env` — this
session cookie expires, so it needs refreshing whenever requests start
getting redirected to the login page.

Writes `output/main/enriched_irap_upload.csv` (irap_upload.csv + star rating
columns appended).

## Step 4 (optional) — `csvwkt2geojson.py`

    uv run python csvwkt2geojson.py

Converts a Star-Rating CSV (needs `Latitude/Longitude start/end` columns) into
a GeoJSON `FeatureCollection` of `LineString` features, for loading into
kepler.gl / QGIS / a dashboard.

**Naming note**: Step 3 writes `output/main/enriched_irap_upload.csv`, but
this script's `IN_CSV` currently points at `output/main/SRS_rating(ViDA_API).csv`
by default — rename/copy Step 3's output to that name first, or edit
`IN_CSV`/`OUT_FILE` at the top of the script to point at whatever CSV you're
actually using.

Also adds a `"<column> text"` human-readable label alongside every coded
iRAP column's numeric value (e.g. `"Area type": 2, "Area type text": "Urban"`)
— dashboard-only, sourced from the iRAP v3.10 Coding Manual's own code
tables, and not written back to the CSV.

## Pipeline summary

```
gather.py  ──▶  output/main/road_attributes_raw.csv
encode.py  ──▶  output/main/irap_upload.csv
extract_irap_srs.py  ──▶  output/main/enriched_irap_upload.csv  (+ Star Ratings)
csvwkt2geojson.py (optional, after renaming the CSV to match IN_CSV)  ──▶  *.geojson
```
