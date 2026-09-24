# Model and Safety Score

Street View object detection and the Speed Safety Score for the ADB AI for Safer
Roads challenge.

## Pipeline

1. Load road segments from `ADB_Innovation_Thailand.geojson`.
2. Match 2024 accident points (speeding / cutting-in) to their nearest segment.
3. Sample 1000 segments with at least two accidents. For each, take the two
   farthest-apart accident points and fetch Google Street View at those exact
   coordinates.
4. Run YOLO-World object detection and Mask2Former scene segmentation on each of
   the 2000 images and write labelled imagery.
5. Compute the per-segment Speed Safety Score and the speed / objects / accidents
   correlation bubble chart.

## Setup

    uv sync
    cp .env.example .env      # add GSV_API_KEY for live Street View

torch and torchvision are pinned to the CUDA 12.8 wheel index (RTX 50-series).

## Run

    uv run python accident_svi.py       # imagery pipeline: points, SVI, YOLO + Mask2Former
    uv run adb-pipeline strata          # per-segment Speed Safety Score
    uv run adb-pipeline accidents       # join 2024 crashes to segments
    uv run adb-pipeline correlate       # object vs speed stats
    uv run adb-pipeline accident-correlate
    uv run adb-pipeline plot            # speed / objects / accidents bubble chart
    uv run adb-pipeline all             # every analysis stage in order

## Layout

- `accident_svi.py`: the Street View imagery pipeline
- `adb_pipeline/`: segment loading, accident join, detection, segmentation,
  safety score, correlation, bubble chart
- `ADB_Innovation_Thailand.geojson`, `accident_data_2024_english.csv`: inputs

Model weights (YOLO-World, Mask2Former) download at runtime and are not committed.
Over/under is judged on the 85th-percentile operating speed vs the posted limit.
Correlations are observational, not causal.

## iRAP v3.10 pilot (25-segment ViDA upload)

Encodes 25 stratified ADB segments into the iRAP v3.10 Drive-on-Left upload
schema. Reference documents live in this folder:

- `iRAP_v3.10_Coding_Manual_Drive_on_Left.pdf` — code enums for each attribute
- `Upload_file_specification_model_v3.10.xlsx` — the 72-column upload schema
  (the encoder's `COLUMNS` list is identical to this file's row-1 headers)

### One-shot run

    uv run python irap_run.py

This chains: sample 25 segments (5 per 50/60/70/80/90 kph bin, Mapillary-covered)
→ fetch 50 Mapillary images (2 per segment) → YOLO-World detection →
Mask2Former scene segmentation → labelled images + `labelled_metadata.csv` →
Overpass API for junctions/signals/sidewalks → encode → `irap_upload_v310.csv`.

All outputs land in `output/_irap_pilot/`. Flags:

    uv run python irap_run.py --sample-only   # stop after sampling
    uv run python irap_run.py --no-encode     # stop after CV pipeline

### Full-network reference (all 11,544 segments, no CV/OSM)

    uv run python generate_irap_reference.py  # -> output/irap_reference_all_11544.csv

Produces a 72-column iRAP-schema CSV for every ADB segment, using real data for
Speed limit / Operating speeds / Land use / Curvature / Number of lanes / AADT
lookup, and documented iRAP defaults for CV/OSM-derived attributes. Reference
artefact for the "Relationship to iRAP" writeup — NOT a valid ViDA upload.

### Upload to ViDA

Log in to ViDA, create a project set to **model v3.10, Drive on Left**, and
upload `output/_irap_pilot/irap_upload_v310.csv`. ViDA returns star ratings +
the Safer Roads Investment Plan.

### What is filled from real data vs. defaulted

- **From ADB geojson**: Speed limit, Operating Speed 85th/mean, Area type,
  Land use (rural/urban only), Curvature (from LineString geometry).
- **From CV (YOLO + Mask2Former on 2 Mapillary images/segment)**: Roadside
  object/distance, Paved shoulder, Delineation, Sidewalk, Cycling facilities,
  Vehicle parking, Crossing facility, Street lighting.
- **From OSM via Overpass API**: Intersection type (with signalisation), Street
  lighting, Sidewalk (tag), Number of lanes (tag), Cycleway (tag).
- **Class-based / country defaults** (flagged in `SRS_pipeline/encode.py`): AADT,
  Motorcycle %, HGV %, plus safe defaults for Grade, Skid resistance,
  Road condition, School zone, Roadworks, etc.

Upgrade cost (col 16) is coded as terrain-difficulty tier (Low/Medium/High);
ViDA applies Thailand's per-km countermeasure costs automatically.

Intersection block follows Coding Manual v3.10 p117 pairing rules: when type=12
(None), quality=3 (Not Applicable), volume=0, channelisation=1 (Not present),
side-road crossing=7 (No facility), crossing quality=3 (Not Applicable).
