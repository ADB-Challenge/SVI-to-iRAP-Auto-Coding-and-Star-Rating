"""Road curvature for IMAGE LABELLING only (not analytics).

Derived purely from a segment's GeoJSON LineString vertices:
  * sinuosity = arc length / straight-line distance (1.0 = straight)
  * sharp_curves_per_km = count of <CURVE_RADIUS_THRESH_M turns per km, found on a
    uniform CURVE_RESAMPLE_M resample so the count is not biased by vertex spacing.

`curvature_by_objectids` returns {OBJECTID: {sinuosity, curves_per_km}} so the
labelled-image stage can print it on each frame.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache

from pyproj import Transformer
from shapely.geometry import LineString

from . import config


@lru_cache(maxsize=None)
def _transformer(epsg: int) -> Transformer:
    # always_xy=True -> we pass/get (lon, lat) / (easting, northing) order.
    return Transformer.from_crs("EPSG:4326", epsg, always_xy=True)


def _utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing this point (N/S hemisphere aware)."""
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _to_metric(coords):
    """Project lon/lat to UTM easting/northing metres via pyproj (proper CRS)."""
    lon0 = sum(c[0] for c in coords) / len(coords)
    lat0 = sum(c[1] for c in coords) / len(coords)
    tr = _transformer(_utm_epsg(lon0, lat0))
    xs, ys = tr.transform([c[0] for c in coords], [c[1] for c in coords])
    return list(zip(xs, ys))


def _menger_radius(p1, p2, p3) -> float:
    a = math.dist(p1, p2)
    b = math.dist(p2, p3)
    c = math.dist(p1, p3)
    area2 = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    if area2 < 1e-9:
        return math.inf
    return (a * b * c) / (2.0 * area2)


def segment_curvature(coords) -> dict:
    """sinuosity + sharp-curves-per-km for one LineString (list of [lon, lat])."""
    line = LineString(_to_metric(coords))
    L = line.length
    D = math.dist(line.coords[0], line.coords[-1])
    sinu = round(min(L / D, config.SINUOSITY_CAP), 2) if D > 1.0 else None

    step = config.CURVE_RESAMPLE_M
    n = max(int(L // step), 2)
    pts = [(p.x, p.y) for p in (line.interpolate(min(i * step, L)) for i in range(n + 1))]
    n_curves, prev = 0, False
    for i in range(1, len(pts) - 1):
        sharp = _menger_radius(pts[i - 1], pts[i], pts[i + 1]) < config.CURVE_RADIUS_THRESH_M
        if sharp and not prev:
            n_curves += 1
        prev = sharp
    km = L / 1000.0
    return {"sinuosity": sinu,
            "curves_per_km": round(n_curves / km, 2) if km > 0 else 0.0}


def curvature_by_objectids(objectids) -> dict:
    """Map {OBJECTID: {sinuosity, curves_per_km}} for the requested segments."""
    wanted = set(int(o) for o in objectids)
    out = {}
    gj = json.load(open(config.GEOJSON_PATH, encoding="utf-8"))
    for f in gj["features"]:
        oid = f["properties"].get("OBJECTID")
        if oid is None or int(oid) not in wanted:
            continue
        c = f["geometry"]["coordinates"]
        if c and isinstance(c[0][0], list):
            c = [pt for part in c for pt in part]
        if len(c) >= 2:
            out[int(oid)] = segment_curvature(c)
    return out
