"""Main pipeline, script 2 of 2 -- encode gather.py's 500m readings into the
72-column iRAP v3.10 upload schema, ONE ROW PER READING (not aggregated by
OBJECTID). Each reading already represents a specific ~500m stretch of road
with its own single image, so it gets its own iRAP row and its own eventual
ViDA Star Rating -- matching genuine iRAP survey practice (short discrete
sections), rather than collapsing an entire 20-50km segment into one row.

OSM excluded (every `osm` dict is `{}`) -- gather.py never collects OSM data
in the first place (dropped entirely; the public Overpass instance proved
unreliable -- either unreachable or 504-timing-out under load), so this is a
hard requirement here, not a choice.

Chainage per row:
    Distance = this reading's own distance_m / 1000 (already real chainage,
               computed by gather.py's haversine walk along the segment's
               full geometry)
    Length   = distance to the NEXT reading in the same OBJECTID (normally
               0.5km, shorter for the reading just before a segment's true
               endpoint) -- the last reading in a segment has no "next," so
               it falls back to the standard 500m step.
    Lat/Lon start/end = this reading's own coordinates -> the next reading's
               coordinates (the road section THIS row actually covers), not
               the whole segment's endpoints.

CV evidence per row comes from that reading's OWN single image only -- no
merging across neighbouring readings, since each row represents one specific
spot, not a pair.

The 3 CV-based inference rules below (Roadworks/Median type/Roadside object
scene-fraction reinforcement) came out of an earlier experiment -- see
RULES.md in this folder for the full rationale, confidence level, and
empirical fire-rates behind each one before trusting their output blindly.

Reads:  output/main/road_attributes_raw.csv
Writes: output/main/irap_upload.csv
        output/main/rule_fire_log.csv
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from adb_pipeline import config

RAW_CSV = config.OUTPUT_DIR / "main" / "road_attributes_raw.csv"
OUT_CSV = config.OUTPUT_DIR / "main" / "irap_upload.csv"
RULE_LOG_CSV = config.OUTPUT_DIR / "main" / "rule_fire_log.csv"

POINT_INTERVAL_M = 500.0  # must match gather.py's own constant

# ---------------------------------------------------------------------------
# Coder metadata
# ---------------------------------------------------------------------------
CODER_EMAIL = "m4060@n-koei.co.jp"
# ViDA validator requires ISO-8601 (YYYY-MM-DD) despite the Excel spec header
# labelling this "date (format dd/mm/yyyy)". Follow the validator.
CODING_DATE = "2026-09-02"
SURVEY_DATE = "2026-09-02"

# ---------------------------------------------------------------------------
# Country/region defaults for fields we have no direct signal for.
# All values are integer iRAP codes from Upload_file_specification_model_v3.10.
# ---------------------------------------------------------------------------
AADT_BY_ROADCLASS = {
    "motorway":  30000,
    "trunk":     15000,
    "primary":    8000,
    "secondary":  3000,
    "tertiary":   1500,
}
AADT_DEFAULT = 3000

MOTORCYCLE_PCT_CODE = 5   # "5 - 11% to 20%"  (Thailand PTW share, approx.)
HGV_PCT_CODE       = 3    # "3 - 1% to 5%"
FATALITY_MULTIPLIER = 1

# Fields we can't infer -> safe defaults (documented in the spec).
CARRIAGEWAY_UNDIVIDED   = 3   # "3 - Undivided road"
UPGRADE_COST_LOW        = 1   # "1 - Low"
UPGRADE_COST_MEDIUM     = 2   # urban -> Medium (higher land value)
VARIABLE_SPEED_ABSENT   = 1   # "1 - Not present"
SPEED_DIFF_ABSENT       = 1   # iRAP field = per-vehicle-class differential, not F85-SL
CENTRELINE_RUMBLE_NONE  = 1
SHOULDER_RUMBLE_NONE    = 1
MEDIAN_CENTRELINE       = 11  # undivided road: centreline only (fallback when no barrier evidence)
INTERSECTION_CHAN_NONE  = 1
INTERSECTION_QUAL_NA    = 3   # "3 - Not applicable"
INTERSECTION_QUAL_OK    = 1   # "1 - Adequate"
PROPERTY_ACCESS_NONE    = 4
PROPERTY_ACCESS_RES     = 3   # "3 - Residential Access 1 to 2" (used for urban)
GRADE_FLAT              = 1   # no DEM data
ROAD_CONDITION_MEDIUM   = 2
SKID_ADEQUATE           = 1
PED_CHAN_NONE           = 1
SPEED_MGMT_NONE         = 1
SCHOOL_ZONE_NA          = 4
SCHOOL_SUP_NA           = 3
SERVICE_ROAD_NONE       = 1
PTW_NONE                = 6
ROADWORKS_NONE          = 1
INCIDENT_WARNING_NONE   = 1   # "1 - Not present"
SPEED_CAMERA_NONE       = 1   # "1 - No speed camera"
SIGHT_DIST_ADEQUATE     = 1
PED_FLOW_ZERO           = 1
PED_FLOW_LOW            = 2   # "1 to 5"; used when urban
BICYCLE_FLOW_ZERO       = 1

# ---------------------------------------------------------------------------
# The exact 72-column header row from Upload_file_specification_model_v3.10.xlsx
# ---------------------------------------------------------------------------
COLUMNS = [
    "Coder name", "Coding date", "Road survey date", "Image reference",
    "Road Name", "Section", "Distance", "Length",
    "Latitude start", "Longitude start", "Latitude end", "Longitude end",
    "Landmark", "Comments", "Carriageway label", "Upgrade cost",
    "Land use - drivers side", "Land use - passenger side", "Area type",
    "Speed limit", "Variable speed limit", "Speed differential",
    "Median type", "Centreline rumble strips",
    "Roadside severity - drivers side distance",
    "Roadside severity - drivers side object",
    "Roadside severity - passenger side distance",
    "Roadside severity - passenger side object",
    "Shoulder rumble strips",
    "Paved shoulder - drivers side", "Paved shoulder - passenger side",
    "Intersection type", "Intersection channelisation",
    "Intersecting road volume", "Intersection quality",
    "Property access points", "Number of lanes", "Lane width",
    "Curvature", "Quality of curve", "Grade", "Road condition",
    "Skid resistance", "Delineation", "Street lighting",
    "Crossing facility - inspected road", "Crossing quality",
    "Crossing facility - side road", "Pedestrian channelisation ",
    "Speed management", "Vehicle parking",
    "Sidewalk - drivers side", "Sidewalk - passenger side",
    "School zone warning", "School zone crossing supervisor",
    "Service road", "PTW facilities", "Cycling facilities",
    "Roadworks", "Sight distance", "Vehicle flow (AADT)",
    "Motorcycle %", "HGV %",
    "Pedestrian peak hour flow across the road",
    "Pedestrian peak hour flow along the road driver-side",
    "Pedestrian peak hour flow along the road passenger-side",
    "Bicycle peak hourly flow",
    "Operating Speed (85th percentile)", "Operating Speed (mean)",
    "Incident detection warning system", "Speed cameras",
    "Annual Fatality Growth Multiplier",
]
assert len(COLUMNS) == 72, f"expected 72 columns, got {len(COLUMNS)}"


# ---------------------------------------------------------------------------
# CV parsing helpers
# ---------------------------------------------------------------------------
_TREE_RE   = re.compile(r"\b(tree|palm|shrub)\b", re.I)
_POLE_RE   = re.compile(r"\b(pole|post|sign|streetlight|street lamp|lamppost|traffic light|"
                         r"utility pole|telephone pole|pillar|column)\b", re.I)
_BUILDING_RE = re.compile(r"\b(building|house|wall)\b", re.I)
_FENCE_RE  = re.compile(r"\b(fence|railing|hedge)\b", re.I)
_BARRIER_RE = re.compile(r"\b(guard ?rail|guardrail|barrier|jersey|concrete barrier)\b", re.I)
_CAR_PARKED_RE = re.compile(r"\b(parked)\b", re.I)


def _parse_detections(s) -> list[str]:
    """detected_objects string ('label:score;label:score;...') -> [labels]."""
    if not isinstance(s, str) or not s:
        return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        label = part.split(":", 1)[0].strip()
        if label:
            out.append(label)
    return out


def _parse_scene(s) -> dict[str, float]:
    """scene_composition string ('name:pct%;name:pct%;...') -> {name: fraction}."""
    if not isinstance(s, str) or not s:
        return {}
    out = {}
    for part in s.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, val = part.rsplit(":", 1)
        val = val.strip().rstrip("%")
        try:
            out[name.strip()] = float(val) / 100.0
        except ValueError:
            continue
    return out


def _aggregate_cv(meta_df: pd.DataFrame) -> dict:
    """Union detected object labels; average scene fractions. Called with a
    single-row group in this encoder (one reading = one image), so this
    naturally reduces to "just that image's own values" -- no merging across
    neighbouring readings like the old per-segment encoder did."""
    labels = Counter()
    for s in meta_df["detected_objects"].fillna(""):
        for lb in _parse_detections(s):
            labels[lb] += 1
    scene_sum: dict[str, float] = {}
    n = 0
    for s in meta_df["scene_composition"].fillna(""):
        d = _parse_scene(s)
        if not d:
            continue
        n += 1
        for name, f in d.items():
            scene_sum[name] = scene_sum.get(name, 0.0) + f
    scene_avg = {k: v / n for k, v in scene_sum.items()} if n else {}
    return {"labels": labels, "scene": scene_avg, "n_images": len(meta_df)}


def _image_refs(grp: pd.DataFrame) -> str:
    if "image_id" not in grp.columns:
        return ""
    ids = []
    for iid in grp["image_id"].dropna():
        try:
            ids.append(f"https://www.mapillary.com/app/?pKey={int(iid)}")
        except (TypeError, ValueError):
            continue
    return ";".join(ids[:2])


# ---------------------------------------------------------------------------
# Per-column encoders (return the raw code -- integer or numeric).
# ---------------------------------------------------------------------------
def enc_carriageway_label(seg) -> int:
    return CARRIAGEWAY_UNDIVIDED


def enc_upgrade_cost(seg) -> int:
    return UPGRADE_COST_MEDIUM if str(seg.get("LandUse", "")).upper() == "URBAN" else UPGRADE_COST_LOW


def enc_land_use(seg) -> int:
    lu = str(seg.get("LandUse", "")).upper()
    if lu == "URBAN":
        return 3       # Residential (broadest urban default)
    return 1           # Undeveloped areas (rural / None fallback)


def enc_area_type(seg) -> int:
    return 2 if str(seg.get("LandUse", "")).upper() == "URBAN" else 1


def enc_speed_limit(seg) -> int:
    try:
        return int(round(float(seg["SpeedLimit"])))
    except (TypeError, ValueError, KeyError):
        return 50


def enc_roadside_distance(cv) -> int:
    """We can't measure real metres from a single image -- proxy: if there is
    any close object detected, assume 1-<5m; otherwise assume >=10m."""
    lbls = " ".join(cv["labels"].keys())
    if (_TREE_RE.search(lbls) or _POLE_RE.search(lbls) or _BUILDING_RE.search(lbls)
        or _FENCE_RE.search(lbls) or _BARRIER_RE.search(lbls)):
        return 2    # 1 to <5m
    return 4        # >=10m


def enc_paved_shoulder(cv, seg) -> int:
    """Use Mask2Former road/service-lane fraction as a proxy for shoulder presence."""
    road_frac = cv["scene"].get("road", 0) + cv["scene"].get("service lane", 0)
    if road_frac >= 0.35:
        return 2   # Medium 1 to <2.4m
    if road_frac >= 0.20:
        return 3   # Narrow 0 to <1m
    return 4       # None


def enc_intersection_type(osm) -> int:
    hint = str(osm.get("intersection_hint", "none"))
    sig = bool(osm.get("signalised", False))
    if hint == "4leg":
        return 10 if sig else 8    # 4+ leg signalised / plain
    if hint == "3leg":
        return 6 if sig else 4     # 3 leg signalised / plain
    return 12                       # None


def enc_intersection_channelisation(osm) -> int:
    return INTERSECTION_CHAN_NONE   # not enough signal from OSM alone


def enc_intersecting_road_volume(osm) -> int:
    """Coding Manual v3.10 p117: when Intersection type = 12 (None), volume
    MUST be 0 (not blank). Otherwise, crude side-road AADT default."""
    if str(osm.get("intersection_hint", "none")) == "none":
        return 0
    return 1000


def enc_intersection_quality(osm) -> int:
    """Coding Manual v3.10 p117: when Intersection type = 12 (None), quality
    MUST be '3 - Not Applicable' (NOT blank -- the ViDA error text
    'must not have quality' is misleading; the manual is authoritative)."""
    if str(osm.get("intersection_hint", "none")) == "none":
        return INTERSECTION_QUAL_NA
    return INTERSECTION_QUAL_OK


def enc_number_of_lanes(seg, osm) -> int:
    if osm.get("osm_lanes"):
        try:
            n = int(osm["osm_lanes"])
            if n <= 1: return 1
            if n == 2: return 2
            if n == 3: return 3
            return 4
        except (TypeError, ValueError):
            pass
    rc = str(seg.get("RoadClass", "")).lower()
    if rc == "motorway": return 4
    if rc == "trunk":    return 3
    return 2


def enc_lane_width(seg) -> int:
    rc = str(seg.get("RoadClass", "")).lower()
    if rc in {"motorway", "trunk"}: return 1
    return 2


def enc_curvature(curv_stats) -> int:
    cpk = float((curv_stats or {}).get("curves_per_km") or 0.0)
    if cpk <= 0.1: return 1
    if cpk <= 1.0: return 2
    if cpk <= 3.0: return 3
    return 4


def enc_quality_of_curve(curv_stats) -> int:
    return 3 if enc_curvature(curv_stats) == 1 else 1


def enc_delineation(cv) -> int:
    """Adequate if Mask2Former sees lane markings; else Poor."""
    if cv["scene"].get("lane marking", 0) >= 0.005:
        return 1
    return 2


def enc_street_lighting(cv, osm) -> int:
    """Present if OSM says the way is lit / has street_lamp nodes, OR if YOLO
    detected any streetlight/lamppost across the segment's images."""
    if bool(osm.get("street_lit", False)):
        return 2
    for k, v in cv["labels"].items():
        if re.search(r"\b(streetlight|street lamp|lamppost|lamp post)\b", k, re.I) and v > 0:
            return 2
    return 1


def enc_crossing_inspected(cv, osm) -> int:
    scene = cv["scene"]
    has_crosswalk = scene.get("crosswalk marking", 0) >= 0.001 or int(osm.get("n_crossings", 0)) > 0
    if not has_crosswalk:
        return 7    # No facility
    return 3 if bool(osm.get("signalised", False)) else 5   # Signalised / Marked


def enc_crossing_side_road(cv, osm) -> int:
    """Coding Manual v3.10 p117: when Intersection type = 12 (None),
    Crossing Facility on Side Road MUST be '7 - No facility'."""
    if str(osm.get("intersection_hint", "none")) == "none":
        return 7
    return enc_crossing_inspected(cv, osm)


def enc_crossing_quality(cv, osm) -> int:
    """Coding Manual v3.10 p117: when NEITHER the inspected road NOR the side
    road has a crossing facility, Crossing Quality MUST be Not Applicable (3)."""
    inspected = enc_crossing_inspected(cv, osm)
    side = enc_crossing_side_road(cv, osm)
    if inspected == 7 and side == 7:
        return 3
    return 1


def enc_vehicle_parking(cv) -> int:
    """Heuristic: many parked-looking labels -> 'One side' / 'Two sides'."""
    lbls = cv["labels"]
    parked = sum(v for k, v in lbls.items() if _CAR_PARKED_RE.search(k))
    return 2 if parked >= 3 else 1


def enc_sidewalk(cv, osm) -> int:
    """OSM sidewalk tag wins if present; else use Mask2Former sidewalk fraction."""
    tag = str(osm.get("osm_sidewalk") or "").lower()
    if tag in {"both", "left", "right", "yes", "separate"}:
        return 3   # Sidewalk 1m to <3m from road (typical urban)
    if tag == "no":
        return 5
    sw = cv["scene"].get("sidewalk", 0)
    if sw >= 0.03:
        return 3
    if sw >= 0.01:
        return 4    # 0 to <1m from road
    return 5        # None


def enc_cycling_facilities(cv, osm) -> int:
    cw = str(osm.get("osm_cycleway") or "").lower()
    if cw in {"track", "path"}:
        return 2   # Segregated bicycle path
    if cw in {"lane", "shared_lane", "buffered_lane"}:
        return 3   # Bicycle lane
    return 4       # None


def enc_aadt(seg) -> int:
    rc = str(seg.get("RoadClass", "")).lower()
    return AADT_BY_ROADCLASS.get(rc, AADT_DEFAULT)


def enc_ped_flow(seg) -> int:
    """Urban -> 1-5, Rural -> 0."""
    return PED_FLOW_LOW if str(seg.get("LandUse", "")).upper() == "URBAN" else PED_FLOW_ZERO


def enc_operating_speed(v) -> int | str:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# CV-based inference rules -- see RULES.md for rationale/confidence/evidence.
# ---------------------------------------------------------------------------
def enc_roadworks_cv(cv: dict) -> tuple[int, bool]:
    """Rule 1: 'cone' detected -> Minor roadworks. See RULES.md #1."""
    lbls = " ".join(cv["labels"].keys())
    fired = "cone" in lbls.lower()
    return (2 if fired else ROADWORKS_NONE), fired


def enc_median_type_cv(cv: dict) -> tuple[int, bool]:
    """Rule 2: guard rail (label) or BARRIER (scene) -> metal safety barrier.
    See RULES.md #2 for the median-vs-roadside placement caveat."""
    lbls = " ".join(cv["labels"].keys())
    fired = bool(_BARRIER_RE.search(lbls)) or "BARRIER" in cv["scene"]
    return (1 if fired else MEDIAN_CENTRELINE), fired


def enc_roadside_object_cv(cv: dict) -> tuple[int, str]:
    """Same priority order as a plain label-only roadside-object encoder, but
    ALSO checks Mask2Former scene fractions for pole/utility pole/wall (not
    just YOLO labels) -- rules 3 and 4 in RULES.md."""
    lbls = " ".join(cv["labels"].keys())
    scene = cv["scene"]
    if _BARRIER_RE.search(lbls):
        return 1, "barrier(label)"
    if _TREE_RE.search(lbls):
        return 11, "tree(label)"
    pole_label = bool(_POLE_RE.search(lbls))
    pole_scene = "pole" in scene or "utility pole" in scene
    if pole_label or pole_scene:
        return 12, "pole(label)" if pole_label else "pole(scene)"
    wall_label = bool(_BUILDING_RE.search(lbls)) or scene.get("building", 0) > 0.05
    wall_scene = "wall" in scene
    if wall_label or wall_scene:
        return 13, "wall(label)" if wall_label else "wall(scene)"
    if _FENCE_RE.search(lbls) or scene.get("fence", 0) > 0.02:
        return 14, "fence"
    return 17, "none"


def build_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Defensive filter (gather.py should already exclude these upstream) -- a
    # reading with no real fetched image has no CV evidence behind it, so
    # encoding it would just be defaults dressed up as a real scored row.
    # Dropping one also correctly widens the Length of whichever surviving
    # reading now precedes the next gap, since Length is derived from
    # consecutive rows AFTER this filter runs.
    n_before = len(df)
    df = df[df["image_path"].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[encode] dropped {n_dropped}/{n_before} readings with no real "
              f"Mapillary image -> {len(df)} readings remain")

    osm: dict = {}
    rows, fire_log = [], []

    for oid, grp in df.groupby("OBJECTID"):
        grp = grp.sort_values("distance_m").reset_index(drop=True)
        n = len(grp)
        for i in range(n):
            r = grp.iloc[i]
            cv = _aggregate_cv(grp.iloc[[i]])  # this reading's own image only

            distance_km = round(float(r["distance_m"]) / 1000.0, 3)
            if i < n - 1:
                nxt = grp.iloc[i + 1]
                length_km = round((float(nxt["distance_m"]) - float(r["distance_m"])) / 1000.0, 3)
                lat_end, lon_end = float(nxt["lat"]), float(nxt["lon"])
            else:
                length_km = round(POINT_INTERVAL_M / 1000.0, 3)  # no next reading -- fallback
                lat_end, lon_end = float(r["lat"]), float(r["lon"])

            curv_stats = {"curves_per_km": r.get("curves_per_km"), "sinuosity": r.get("sinuosity")}

            roadworks_code, roadworks_fired = enc_roadworks_cv(cv)
            median_code, median_fired = enc_median_type_cv(cv)
            roadside_code, roadside_rule = enc_roadside_object_cv(cv)

            fire_log.append({
                "sample_id": r["sample_id"], "OBJECTID": int(oid), "distance_m": r["distance_m"],
                "roadworks_fired": roadworks_fired, "median_barrier_fired": median_fired,
                "roadside_object_rule": roadside_rule,
                "detected_objects": r.get("detected_objects"),
                "scene_composition": r.get("scene_composition"),
            })

            out = {
                "Coder name": CODER_EMAIL,
                "Coding date": CODING_DATE,
                "Road survey date": SURVEY_DATE,
                "Image reference": _image_refs(grp.iloc[[i]]),
                "Road Name": f"OBJECTID {int(oid)} @ {distance_km:.1f}km",
                "Section": "500m interval, full geometry, OSM excluded",
                "Distance": distance_km,
                "Length": length_km,
                "Latitude start":  float(r["lat"]), "Longitude start": float(r["lon"]),
                "Latitude end":    lat_end,          "Longitude end":   lon_end,
                "Landmark": "",
                "Comments": "",
                "Carriageway label": enc_carriageway_label(r),
                "Upgrade cost": enc_upgrade_cost(r),
                "Land use - drivers side":   enc_land_use(r),
                "Land use - passenger side": enc_land_use(r),
                "Area type": enc_area_type(r),
                "Speed limit": enc_speed_limit(r),
                "Variable speed limit": VARIABLE_SPEED_ABSENT,
                "Speed differential":   SPEED_DIFF_ABSENT,
                "Median type": median_code,
                "Centreline rumble strips": CENTRELINE_RUMBLE_NONE,
                "Roadside severity - drivers side distance":   enc_roadside_distance(cv),
                "Roadside severity - drivers side object":     roadside_code,
                "Roadside severity - passenger side distance": enc_roadside_distance(cv),
                "Roadside severity - passenger side object":   roadside_code,
                "Shoulder rumble strips": SHOULDER_RUMBLE_NONE,
                "Paved shoulder - drivers side":   enc_paved_shoulder(cv, r),
                "Paved shoulder - passenger side": enc_paved_shoulder(cv, r),
                "Intersection type":           enc_intersection_type(osm),
                "Intersection channelisation": enc_intersection_channelisation(osm),
                "Intersecting road volume":    enc_intersecting_road_volume(osm),
                "Intersection quality":        enc_intersection_quality(osm),
                "Property access points": PROPERTY_ACCESS_RES if str(r.get("LandUse", "")).upper() == "URBAN"
                                           else PROPERTY_ACCESS_NONE,
                "Number of lanes": enc_number_of_lanes(r, osm),
                "Lane width":      enc_lane_width(r),
                "Curvature":        enc_curvature(curv_stats),
                "Quality of curve": enc_quality_of_curve(curv_stats),
                "Grade": GRADE_FLAT,
                "Road condition":  ROAD_CONDITION_MEDIUM,
                "Skid resistance": SKID_ADEQUATE,
                "Delineation":     enc_delineation(cv),
                "Street lighting": enc_street_lighting(cv, osm),
                "Crossing facility - inspected road": enc_crossing_inspected(cv, osm),
                "Crossing quality":                   enc_crossing_quality(cv, osm),
                "Crossing facility - side road":      enc_crossing_side_road(cv, osm),
                "Pedestrian channelisation ": PED_CHAN_NONE,
                "Speed management":            SPEED_MGMT_NONE,
                "Vehicle parking":             enc_vehicle_parking(cv),
                "Sidewalk - drivers side":     enc_sidewalk(cv, osm),
                "Sidewalk - passenger side":   enc_sidewalk(cv, osm),
                "School zone warning":         SCHOOL_ZONE_NA,
                "School zone crossing supervisor": SCHOOL_SUP_NA,
                "Service road":     SERVICE_ROAD_NONE,
                "PTW facilities":   PTW_NONE,
                "Cycling facilities": enc_cycling_facilities(cv, osm),
                "Roadworks":        roadworks_code,
                "Sight distance":   SIGHT_DIST_ADEQUATE,
                "Vehicle flow (AADT)": enc_aadt(r),
                "Motorcycle %":   MOTORCYCLE_PCT_CODE,
                "HGV %":          HGV_PCT_CODE,
                "Pedestrian peak hour flow across the road":               enc_ped_flow(r),
                "Pedestrian peak hour flow along the road driver-side":    enc_ped_flow(r),
                "Pedestrian peak hour flow along the road passenger-side": enc_ped_flow(r),
                "Bicycle peak hourly flow": BICYCLE_FLOW_ZERO,
                "Operating Speed (85th percentile)": enc_operating_speed(r.get("F85thPercentileSpeed")),
                "Operating Speed (mean)":            enc_operating_speed(r.get("MedianSpeed")),
                "Incident detection warning system": INCIDENT_WARNING_NONE,
                "Speed cameras":                     SPEED_CAMERA_NONE,
                "Annual Fatality Growth Multiplier": FATALITY_MULTIPLIER,
            }
            rows.append(out)

    return pd.DataFrame(rows, columns=COLUMNS), pd.DataFrame(fire_log)


def main() -> None:
    if not RAW_CSV.exists():
        raise SystemExit(f"[encode] {RAW_CSV} not found -- run gather.py first")
    df = pd.read_csv(RAW_CSV)

    out, fire_log = build_rows(df)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    fire_log.to_csv(RULE_LOG_CSV, index=False)

    print(f"[encode] wrote {OUT_CSV}  ({len(out)} rows x {len(out.columns)} cols)")
    print(f"[encode] wrote {RULE_LOG_CSV}  (per-reading rule trace)")
    n = len(fire_log)
    print(f"[encode] rule fire rates out of {n} readings:")
    print(f"    Roadworks (cone->Minor):        {int(fire_log['roadworks_fired'].sum())}/{n}")
    print(f"    Median type (barrier detected): {int(fire_log['median_barrier_fired'].sum())}/{n}")
    print("    Roadside object rule breakdown:")
    print(fire_log["roadside_object_rule"].value_counts().to_string())
    print("[encode] upload this CSV in ViDA under model v3.10 Drive on Left.")


if __name__ == "__main__":
    main()
