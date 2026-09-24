"""Central configuration: paths, constants, and runtime knobs.

Override any value with an environment variable (or a `.env` file at the repo
root). The only secret you must supply is ``GSV_API_KEY``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root if present.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Inputs / outputs -------------------------------------------------------
GEOJSON_PATH = Path(os.getenv("ADB_GEOJSON", ROOT / "ADB_Innovation_Thailand.geojson"))

ACCIDENT_CSV = Path(os.getenv("ADB_ACCIDENT_CSV", ROOT / "accident_data_2024_english.csv"))

OUTPUT_DIR = Path(os.getenv("ADB_OUTPUT_DIR", ROOT / "output"))

# --- Clean deliverables (top level) ----------------------------------------
IMAGES_DIR = OUTPUT_DIR / "images"             # raw Street View images
LABELLED_DIR = OUTPUT_DIR / "labelled"         # combined YOLO boxes + Mapillary seg
CHARTS_DIR = OUTPUT_DIR / "charts"             # relationship graphs (PNG)
MASTER_CSV = OUTPUT_DIR / "master_table.csv"   # the one consolidated table
SCORECARD_CSV = OUTPUT_DIR / "safety_scorecard.csv"  # per-category safety scorecard

# --- Pipeline intermediates (tucked away under _pipeline) -------------------
PIPELINE_DIR = OUTPUT_DIR / "_pipeline"
SAMPLES_CSV = PIPELINE_DIR / "mismatch_samples.csv"
SAMPLES_GEOJSON = PIPELINE_DIR / "mismatch_samples.geojson"
GSV_METADATA_CSV = PIPELINE_DIR / "gsv_metadata.csv"
DETECTIONS_JSON = PIPELINE_DIR / "detections.json"
LANES_CSV = PIPELINE_DIR / "lanes.csv"
# Per-image Mask2Former scene-composition fractions (road/vegetation/building/
# etc, whole-image basis -- same numbers shown in each labelled image's own
# legend), keyed by sample_id. See labelled.py.
SCENE_COMPOSITION_JSON = PIPELINE_DIR / "scene_composition.json"
# Post-run summary: one row per successfully-fetched image, pointing at the
# labelled (not raw) image, plus its YOLO detections -- see labelled_metadata.py
LABELLED_METADATA_CSV = PIPELINE_DIR / "labelled_metadata.csv"
CORRELATION_DIR = PIPELINE_DIR / "correlation"
MODEL_DIR = PIPELINE_DIR / "model"
SEGMENT_ACCIDENTS_CSV = PIPELINE_DIR / "segment_accidents.csv"  # crashes joined to segments

# --- Sampling ---------------------------------------------------------------
N_SAMPLES = int(os.getenv("ADB_N_SAMPLES", "1000"))
RANDOM_SEED = int(os.getenv("ADB_SEED", "42"))

# A segment counts as a "mismatch" when |MedianSpeed - SpeedLimit| exceeds this
# many km/h. Keeps trivial rounding noise out of the analysis.
MISMATCH_THRESHOLD_KMH = float(os.getenv("ADB_MISMATCH_THRESHOLD", "5.0"))

# --- Speed-limit categories + safety scoring (assessment framework) ---------
# Segments are grouped by posted speed limit (binned to nearest 10) into these
# categories, then RANDOMLY sampled within each (not over/under-balanced) so the
# observed compliance rate is real.
SPEED_CATEGORIES = [50, 60, 70, 80, 90]           # kph (Category E..A)
CATEGORY_CAP = int(os.getenv("ADB_CATEGORY_CAP", "250"))  # max segments per category

# Speed Safety Score (0-100, higher = LESS safe): blends how MANY exceed the
# limit (prevalence) with how FAR the fast cohort exceeds it (severity).
SAFETY_SEV_CAP = float(os.getenv("ADB_SAFETY_SEV_CAP", "30"))   # km/h over = max severity
SAFETY_W_PREV = float(os.getenv("ADB_SAFETY_W_PREV", "0.5"))    # weight on prevalence
SAFETY_W_SEV = float(os.getenv("ADB_SAFETY_W_SEV", "0.5"))      # weight on severity

# --- Accident data (2024 Thailand crashes) ----------------------------------
# Each crash point attaches to the NEAREST road segment within this many metres
# (else it stays unmatched — reported honestly, not forced onto a distant road).
ACCIDENT_MATCH_RADIUS_M = float(os.getenv("ADB_ACCIDENT_RADIUS_M", "120"))
# Only these two crash causes are kept; every other cause is excluded entirely
# from the join and the correlation. Each kept crash is exactly ONE of the two,
# so the per-segment counts sum to the total.
CAUSE_SPEEDING = "Speeding"
CAUSE_CUTTING = "Pedestrian/vehicle/animal cutting in front suddenly"
ACCIDENT_KEEP_CAUSES = {CAUSE_SPEEDING, CAUSE_CUTTING}

# --- Road curvature (from the GeoJSON LineString geometry only) --------------
# Curve detection resamples each line to a uniform spacing (metres) so the count
# is not biased by how finely a road was digitised, then flags points whose local
# turning radius is below the threshold as "sharp curves".
CURVE_RESAMPLE_M = float(os.getenv("ADB_CURVE_RESAMPLE_M", "50"))
CURVE_RADIUS_THRESH_M = float(os.getenv("ADB_CURVE_RADIUS_M", "200"))
# Sinuosity (arc length / straight distance) is unbounded for loop-like segments
# (start ~ end), so it is clipped to this maximum.
SINUOSITY_CAP = float(os.getenv("ADB_SINUOSITY_CAP", "5.0"))

# --- Google Street View Static API ------------------------------------------
GSV_API_KEY = os.getenv("GSV_API_KEY", "")
GSV_IMAGE_SIZE = os.getenv("GSV_IMAGE_SIZE", "640x640")
GSV_FOV = os.getenv("GSV_FOV", "90")
GSV_PITCH = os.getenv("GSV_PITCH", "0")
# When no key is configured we run in mock mode: synthetic placeholder images so
# the rest of the pipeline (detection, correlation) is fully exercisable.
GSV_MOCK = os.getenv("GSV_MOCK", "").lower() in {"1", "true", "yes"} or not GSV_API_KEY

# --- Mapillary Images API (v4, graph.mapillary.com) — alternative to GSV above
MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "")
# Which pre-rendered thumbnail size to download: thumb_256_url / thumb_1024_url /
# thumb_2048_url / thumb_original_url.
MAPILLARY_IMAGE_FIELD = os.getenv("MAPILLARY_IMAGE_FIELD", "thumb_2048_url")
# Mapillary has no camera control (no heading/fov/pitch like GSV above) — it only
# serves images contributors already captured. We search a small bbox around each
# sampled point and pick the closest existing capture by distance, then heading.
MAPILLARY_SEARCH_RADIUS_M = float(os.getenv("MAPILLARY_SEARCH_RADIUS_M", "50"))
MAPILLARY_MAX_RESULTS = int(os.getenv("MAPILLARY_MAX_RESULTS", "20"))
MAPILLARY_MOCK_IMAGE_SIZE = os.getenv("MAPILLARY_MOCK_IMAGE_SIZE", "640x640")
MAPILLARY_METADATA_CSV = PIPELINE_DIR / "mapillary_metadata.csv"
# When no token is configured we run in mock mode, same idea as GSV_MOCK.
MAPILLARY_MOCK = (os.getenv("MAPILLARY_MOCK", "").lower() in {"1", "true", "yes"}
                   or not MAPILLARY_ACCESS_TOKEN)

# --- Object detection (YOLO-World, promptless open vocabulary) ---------------
# We do NOT hand-pick a class prompt. YOLO-World is loaded with its full
# vocabulary so it detects whatever it recognises in each image.
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8x-worldv2.pt")
# Vocabulary the detector is allowed to recognise:
#   "lvis" -> 1,203 everyday-object categories (the broadest practical setting)
#   "coco" ->    80 common categories (faster, road-traffic focused)
YOLO_VOCAB = os.getenv("YOLO_VOCAB", "lvis").lower()
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.30"))   # higher = fewer hallucinations
# NOTE: YOLO-World/LVIS only detects discrete "things". Background "stuff"
# (grass, sky, road, walls, barriers, buildings) is captured by the semantic
# segmentation scene features in stage6_lanes instead.
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.50"))
YOLO_MAX_DET = int(os.getenv("YOLO_MAX_DET", "300"))
YOLO_BATCH = int(os.getenv("YOLO_BATCH", "16"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "0")  # "0" = first GPU; "cpu" to force CPU

# Exclude on-road vehicles (current traffic) from detection — the analysis is
# about the road *environment*, not the cars passing through it. Matched as whole
# words against each (verbose, multi-synonym) LVIS label.
EXCLUDE_VEHICLES = os.getenv("ADB_EXCLUDE_VEHICLES", "true").lower() in {"1", "true", "yes"}
VEHICLE_KEYWORDS = [
    "car", "automobile", "auto", "truck", "lorry", "bus", "minibus", "coach",
    "motorcycle", "motorbike", "moped", "scooter", "jeep", "van", "minivan",
    "suv", "taxi", "cab", "convertible", "pickup", "ambulance", "fire engine",
    "fire truck", "tow truck", "tractor", "trailer", "semi truck", "limousine",
    "race car", "sports car", "police cruiser", "police van", "school bus",
    "bicycle", "bike", "tricycle", "rickshaw", "tuk-tuk", "go-kart", "golfcart",
    "garbage truck", "dump truck", "trucking rig", "articulated lorry",
]

# Specific noise object classes to drop (matched on the first LVIS synonym, so
# e.g. "can" drops "can/tin can" but NOT "trash can").
OBJECT_EXCLUDE = {"weathervane", "can", "bridge", "helmet", "cap", "ottoman"}

# Extra open-vocabulary prompts appended to YOLO-World so it also boxes elevated
# mass-transit / overpass structures (MRT/skytrain pillars and guideways).
YOLO_EXTRA_CLASSES = [
    "pillar", "overpass", "viaduct",
    # NOTE: "bridge" removed (excluded per request); "elevated road"/"elevated
    # railway"/"column" removed too — too generic, they false-fired on most
    # images. Mapillary "Bridge" (bridge_frac) is the reliable elevated signal.
]

# In the correlation stage, ignore objects seen in fewer than this many sampled
# images (1 = keep absolutely everything). Raise it to suppress noise.
CORR_MIN_SUPPORT = int(os.getenv("ADB_CORR_MIN_SUPPORT", "1"))

# --- Stage 5: relationship model --------------------------------------------
# Object classes become model features only if seen in >= this many images
# (drops one-off detection noise that can't support a learned relationship).
MODEL_MIN_SUPPORT = int(os.getenv("ADB_MODEL_MIN_SUPPORT", "10"))
MODEL_CV_FOLDS = int(os.getenv("ADB_MODEL_CV_FOLDS", "5"))
MODEL_NN_EPOCHS = int(os.getenv("ADB_MODEL_NN_EPOCHS", "120"))
MODEL_NN_DEVICE = os.getenv("MODEL_NN_DEVICE", "cuda")  # falls back to cpu
# Speed-derived fields that must NEVER be model inputs (they leak the target).
LEAKAGE_FIELDS = [
    "MedianSpeed", "F85thPercentileSpeed", "PercentOverLimit", "NumberOverLimit",
    "speed_delta", "abs_delta", "pct_over_limit", "mismatch_direction",
    "Percentile", "RankedPercentile", "InvPercentile", "PercentileBand",
]


def ensure_dirs() -> None:
    """Create all output directories (idempotent)."""
    for d in (OUTPUT_DIR, IMAGES_DIR, LABELLED_DIR, CHARTS_DIR, PIPELINE_DIR,
              CORRELATION_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
