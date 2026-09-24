"""Main pipeline, script 1 of 2 -- gather road attributes for ADB segments
selected by the geojson's own `Percentile` column, sampling a reading every
POINT_INTERVAL_M metres along each segment's FULL drawn geometry (start
lat/lon to end lat/lon).

Why "full geometry": the ADB geojson carries TWO different lengths per
segment -- the drawn LineString ("Shape_Length", e.g. 36.3km for OBJECTID 43)
and a much shorter "RoadLength" property (12.6km for the same segment) that's
probably the stretch the traffic-flow Percentile score was actually computed
over. There's no field in the data that says WHERE within the longer
geometry that shorter stretch falls (checked StreetImageLink's bbox too --
it spans close to the full geometry, doesn't narrow it down). Rather than
guess, this walks the full geometry and documents that caveat here.

No OSM/Overpass step. The public Overpass instance proved unreliable
throughout development -- either unreachable or 504-timing-out under load --
so this pipeline skips it entirely rather than fight it. Every OSM-dependent
encoder in encode.py runs on `osm={}` as a result.

Output: output/main/road_attributes_raw.csv
"""

from __future__ import annotations

import argparse
import json
import math
import time

import pandas as pd
import requests
from shapely.geometry import LineString

from adb_pipeline import config, curvature

# --- redirect outputs to OUR OWN folder --------------------------------------
EXP = config.OUTPUT_DIR / "main"
config.IMAGES_DIR = EXP / "images"
config.LABELLED_DIR = EXP / "labelled"
config.MAPILLARY_METADATA_CSV = EXP / "mapillary_metadata.csv"
config.GSV_METADATA_CSV = EXP / "gsv_metadata.csv"
config.DETECTIONS_JSON = EXP / "detections.json"
config.SAMPLES_CSV = EXP / "samples.csv"
config.SCENE_COMPOSITION_JSON = EXP / "scene_composition.json"
config.LABELLED_METADATA_CSV = EXP / "labelled_metadata.csv"
for d in (EXP, config.IMAGES_DIR, config.LABELLED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- selection / sampling knobs ----------------------------------------------
PERCENTILE_MIN = 0.75
PERCENTILE_MAX = 1.0
POINT_INTERVAL_M = 500.0   # a reading every 500m, walking the FULL geometry
# Safety cap: a broad Percentile range can match hundreds/thousands of
# segments, each costing many Mapillary fetches + YOLO/Mask2Former inference
# calls (roughly geometry_length_km / 0.5 readings PER segment). Raise/lower
# freely, but be aware of the multiplier before raising it much.
MAX_SEGMENTS = 50

_M_PER_DEG_LAT = 110_574.0
_EARTH_R_KM = 6371.0088


# --- Mapillary coverage check -------------------------------------------------
def _mapillary_bbox(lat: float, lon: float, radius_m: float) -> str:
    coslat = math.cos(math.radians(lat))
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / (_M_PER_DEG_LAT * max(coslat, 1e-6))
    return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"


def _mapillary_has_image(lat: float, lon: float, attempts: int = 3) -> bool:
    """True if the Images endpoint reports coverage near (lat, lon).

    The endpoint's bbox search is flaky for sparse (e.g. motorway) coverage --
    the exact same query can alternate between empty and non-empty results a
    few seconds apart (confirmed by hand: 6 identical requests in a row for
    one point came back empty/empty/hit/hit/hit/empty). So we retry a few
    times and count it as covered on the first hit, only giving up after
    every attempt comes back empty.
    """
    if config.MAPILLARY_MOCK:
        return True
    last_err = None
    for i in range(attempts):
        try:
            r = requests.get(
                "https://graph.mapillary.com/images",
                params={
                    "access_token": config.MAPILLARY_ACCESS_TOKEN,
                    "fields": "id",
                    "bbox": _mapillary_bbox(lat, lon, config.MAPILLARY_SEARCH_RADIUS_M),
                    "limit": 1,
                },
                timeout=15,
            )
            r.raise_for_status()
            if len(r.json().get("data", [])) > 0:
                return True
        except Exception as e:
            last_err = e
        if i < attempts - 1:
            time.sleep(1.0)
    if last_err is not None:
        print(f"[gather] Mapillary check failed at {lat:.5f},{lon:.5f}: {last_err}")
    return False


def _load_candidates() -> pd.DataFrame:
    """Every ADB segment with AnalysisStatus == 'Valid' and Percentile in
    [PERCENTILE_MIN, PERCENTILE_MAX]."""
    gj = json.load(open(config.GEOJSON_PATH, encoding="utf-8"))
    rows = []
    for f in gj["features"]:
        p = f["properties"]
        if p.get("AnalysisStatus") != "Valid":
            continue
        pct = p.get("Percentile")
        if pct is None or not (PERCENTILE_MIN <= pct <= PERCENTILE_MAX):
            continue
        coords = f["geometry"]["coordinates"]
        if coords and isinstance(coords[0][0], list):
            coords = [pt for part in coords for pt in part]
        if len(coords) < 2:
            continue
        rows.append({
            "OBJECTID": int(p["OBJECTID"]), "Percentile": pct,
            "SpeedLimit": p.get("SpeedLimit"), "RoadClass": p.get("RoadClass"),
            "F85thPercentileSpeed": p.get("F85thPercentileSpeed"),
            "MedianSpeed": p.get("MedianSpeed"),
            "PercentOverLimit": p.get("PercentOverLimit"),
            "LandUse": p.get("LandUse"), "coords": coords,
        })
    return pd.DataFrame(rows)


# --- distance-based walking along the full geometry --------------------------
def _cumulative_km(coords: list) -> list[float]:
    """Cumulative real-world (haversine) distance in km at each vertex,
    index-aligned with coords. NOT the same as LineString.length*constant --
    that approximation breaks down for paths with significant east-west
    travel (longitude degrees shrink with latitude)."""
    dists = [0.0]
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][:2]
        lon2, lat2 = coords[i + 1][:2]
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
        dists.append(dists[-1] + 2 * _EARTH_R_KM * math.asin(math.sqrt(a)))
    return dists


def _point_at_km(coords: list, cum_km: list[float], target_km: float) -> tuple[float, float, float]:
    """(lat, lon, heading) at target_km along the path, linearly interpolated
    between the two bracketing vertices."""
    for i in range(len(cum_km) - 1):
        if cum_km[i] <= target_km <= cum_km[i + 1]:
            seg_len = cum_km[i + 1] - cum_km[i]
            frac = 0.0 if seg_len == 0 else (target_km - cum_km[i]) / seg_len
            lon1, lat1 = coords[i][:2]
            lon2, lat2 = coords[i + 1][:2]
            lat = lat1 + frac * (lat2 - lat1)
            lon = lon1 + frac * (lon2 - lon1)
            heading = math.degrees(math.atan2(lon2 - lon1, lat2 - lat1)) % 360
            return round(lat, 7), round(lon, 7), round(heading, 1)
    lon, lat = coords[-1][:2]
    return round(lat, 7), round(lon, 7), 0.0


def _readings_along(coords: list) -> list[tuple[float, float, float, float]]:
    """(lat, lon, heading, distance_m) every POINT_INTERVAL_M, from the
    segment's start lat/lon to its end lat/lon -- distance_m is a real
    cumulative chainage within this segment."""
    cum_km = _cumulative_km(coords)
    total_km = cum_km[-1]
    out = []
    d_m = 0.0
    while d_m <= total_km * 1000:
        lat, lon, heading = _point_at_km(coords, cum_km, d_m / 1000)
        out.append((lat, lon, heading, round(d_m, 1)))
        d_m += POINT_INTERVAL_M
    # always include the true end point, even if it doesn't land on a clean
    # 500m multiple
    if out[-1][3] < total_km * 1000 - 1:
        lat, lon, heading = _point_at_km(coords, cum_km, total_km)
        out.append((lat, lon, heading, round(total_km * 1000, 1)))
    return out


def build_points() -> pd.DataFrame:
    cand = _load_candidates()
    print(f"[gather] segments with {PERCENTILE_MIN}<=Percentile<={PERCENTILE_MAX} "
          f"(AnalysisStatus=Valid): {len(cand)}")
    if len(cand) > MAX_SEGMENTS:
        cand = cand.sample(n=MAX_SEGMENTS, random_state=config.RANDOM_SEED)
        print(f"[gather] capped to {MAX_SEGMENTS} segments (MAX_SEGMENTS in this "
              "file) -- raise/lower to change")

    rows, pid, kept = [], 0, 0
    for _, seg in cand.iterrows():
        mid_line = LineString([(c[0], c[1]) for c in seg["coords"]])
        mid = mid_line.interpolate(0.5, normalized=True)
        if not _mapillary_has_image(mid.y, mid.x):
            continue
        kept += 1
        readings = _readings_along(seg["coords"])
        for lat, lon, heading, distance_m in readings:
            rows.append({
                "sample_id": f"F{pid:05d}",
                "OBJECTID": int(seg["OBJECTID"]),
                "distance_m": distance_m,   # real chainage within this segment
                "lat": lat, "lon": lon, "heading": heading,
                "Percentile": seg["Percentile"],
                "SpeedLimit": seg["SpeedLimit"], "RoadClass": seg["RoadClass"],
                "F85thPercentileSpeed": seg["F85thPercentileSpeed"],
                "MedianSpeed": seg["MedianSpeed"],
                "PercentOverLimit": seg["PercentOverLimit"],
                "LandUse": seg["LandUse"],
                # labelled.py (shared stage) requires this column unconditionally.
                "mismatch_direction": "over" if (float(seg["F85thPercentileSpeed"] or 0)
                                                 - float(seg["SpeedLimit"] or 0)) > 0
                                      else "under",
            })
            pid += 1
        print(f"[gather]   OBJECTID {int(seg['OBJECTID'])}: {len(readings)} readings "
              f"over {readings[-1][3]/1000:.1f}km")
    print(f"[gather] kept {kept}/{len(cand)} segments with Mapillary coverage "
          f"-> {len(rows)} total readings (every {POINT_INTERVAL_M:.0f}m)")
    return pd.DataFrame(rows)


def _coverage_report(df: pd.DataFrame) -> None:
    print("[gather] column coverage (non-null / total):")
    for col in df.columns:
        n_ok = int(df[col].notna().sum())
        print(f"    {col:<28} {n_ok}/{len(df)}")


def run(fetch: bool = True, use_cache: bool = False) -> pd.DataFrame:
    if use_cache and config.SAMPLES_CSV.exists() and config.LABELLED_METADATA_CSV.exists():
        print(f"[gather] --use-cache: reusing existing {config.SAMPLES_CSV} and "
              f"{config.LABELLED_METADATA_CSV} -- no Mapillary API calls, no "
              "re-running YOLO/Mask2Former")
        samples = pd.read_csv(config.SAMPLES_CSV)
    else:
        samples = build_points()
        samples.to_csv(config.SAMPLES_CSV, index=False)
        print(f"[gather] wrote {config.SAMPLES_CSV}  ({len(samples)} points)")

    df = samples

    if fetch:
        if use_cache and config.LABELLED_METADATA_CSV.exists():
            print(f"[gather] --use-cache: found {config.LABELLED_METADATA_CSV}, "
                  "skipping Mapillary fetch + detection stages")
        else:
            from adb_pipeline import stage2_mapillary, stage3_detect, labelled, labelled_metadata
            labelled.LABELLED_DIR = config.LABELLED_DIR
            fetch_log = stage2_mapillary.run(samples)
            detections = stage3_detect.run(fetch_log)
            labelled.run(detections, samples)
            labelled_metadata.run()

        meta = pd.read_csv(config.LABELLED_METADATA_CSV)
        cv_cols = ["sample_id", "detected_objects", "n_objects", "scene_composition",
                   "image_path", "image_id", "quality_score", "camera_type", "is_pano",
                   "reprojected"]
        cv_cols = [c for c in cv_cols if c in meta.columns]
        meta_cv = meta[cv_cols].rename(columns={"quality_score": "mapillary_quality_score"})
        # inner join, not left -- meta_cv only has rows with a real fetched
        # image (labelled_metadata.py already excludes ZERO_RESULTS/ERROR),
        # so this drops readings with no real SVI instead of keeping them
        # with blank CV columns.
        n_before = len(df)
        df = df.merge(meta_cv, on="sample_id", how="inner")
        n_dropped = n_before - len(df)
        print(f"[gather] dropped {n_dropped}/{n_before} readings with no real "
              f"Mapillary image -> {len(df)} readings remain")
    else:
        print("[gather] --sample-only: skipping fetch/detect/CV")

    oids = samples["OBJECTID"].unique().tolist()
    curv = curvature.curvature_by_objectids(oids)
    df["sinuosity"] = df["OBJECTID"].map(lambda o: curv.get(int(o), {}).get("sinuosity"))
    df["curves_per_km"] = df["OBJECTID"].map(lambda o: curv.get(int(o), {}).get("curves_per_km"))

    dest = EXP / "road_attributes_raw.csv"
    df.to_csv(dest, index=False)
    print(f"[gather] wrote {dest}  ({len(df)} rows x {len(df.columns)} cols)")
    _coverage_report(df)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-only", action="store_true",
                         help="stop after sampling (skip fetch/detect/CV) -- "
                              "use this first to see the point count before "
                              "committing to the full fetch+CV run")
    parser.add_argument("--use-cache", action="store_true",
                         help="if samples.csv and labelled_metadata.csv already "
                              "exist from a prior run, reuse them instead of "
                              "re-calling the Mapillary API / re-running "
                              "YOLO/Mask2Former -- just rebuilds "
                              "road_attributes_raw.csv from what's already on disk")
    args = parser.parse_args(argv)
    run(fetch=not args.sample_only, use_cache=args.use_cache)


if __name__ == "__main__":
    main()
