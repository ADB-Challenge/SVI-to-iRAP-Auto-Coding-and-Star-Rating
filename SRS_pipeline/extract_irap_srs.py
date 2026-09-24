"""
iRAP ViDA Demonstrator API client
----------------------------------
Sends road segment attributes to https://demonstrator.vida.irap.org/model/4
and extracts the resulting Star Ratings (car, motorcycle, pedestrian, bicycle).

USAGE:
    python irap_demonstrator_client.py

AUTH:
- The Demonstrator is a Laravel app behind PHP session auth + CSRF protection.
- This script:
    1. Uses your PHPSESSID cookie (paste it into SESSION_COOKIE below) to
       authenticate as your logged-in session.
    2. GETs the Demonstrator page fresh each run to read the current
       CSRF token out of the <meta name="csrf-token" content="..."> tag
       (Laravel rotates/validates this per session, so hardcoding a token
       value directly tends to go stale - re-extracting it is more robust).
    3. POSTs to /model/2 with both the session cookie and the token in the
       X-CSRF-TOKEN header, which is Laravel's expected header name for this
       pattern.
- Your PHPSESSID cookie itself will still expire (PHP session timeout, or
  logout) - if you start getting redirected to the login page again, log
  into the Demonstrator in your browser again and copy a fresh PHPSESSID
  from DevTools > Application > Cookies > demonstrator.vida.irap.org.

NOTES:
- Field names below are inferred from the numbered list you posted. If the
  POST fails with a validation error, open DevTools > Network > model/4 >
  Payload / Request tab and copy the *exact* JSON key names ViDA sends -
  paste them into ATTRIBUTES below, replacing my inferred keys.
"""

import os
import re
import requests
import pandas as pd

from adb_pipeline import config

# Set VIDA_PHPSESSID in .env to the FULL PHPSESSID value from DevTools >
# Application > Cookies > demonstrator.vida.irap.org (the "de..." domain one,
# not "vi..."). config (imported above) loads .env.
SESSION_COOKIE_VALUE = os.getenv("VIDA_PHPSESSID", "")
CSV_PATH = config.OUTPUT_DIR / "main" / "irap_upload.csv"

DEMONSTRATOR_PAGE_URL = "https://demonstrator.vida.irap.org/"  # adjust if the actual page URL differs
DEMONSTRATOR_VALIDATE_URL = "https://demonstrator.vida.irap.org/segment/validate/4"
DEMONSTRATOR_API_URL = "https://demonstrator.vida.irap.org/model/4"

# Map road attributes (column names) from csv to API parameter names
COLUMN_MAPPING = {
    "Skid resistance": "skid_resistance",
    "Speed differential": "speed_differential",
    "Crossing facility - inspected road": "crossing_facility_inspected_road",
    "Crossing quality": "crossing_quality",
    "Crossing facility - side road": "crossing_facility_side_road",
    "PTW facilities": "ptw_facilities",
    "Cycling facilities": "cycling_facilities",
    "Operating Speed (mean)": "operating_speed_mean",
    "Carriageway label": "carriageway",
    "Area type": "area_type",
    "Land use - drivers side": "land_use_driver_side",
    "Land use - passenger side": "land_use_passenger_side",
    "Upgrade cost": "upgrade_cost",
    "Roadside severity - drivers side distance": "roadside_severity_driver_side_distance",
    "Roadside severity - drivers side object": "roadside_severity_driver_side_object",
    "Roadside severity - passenger side distance": "roadside_severity_passenger_side_distance",
    "Roadside severity - passenger side object": "roadside_severity_passenger_side_object",
    "Paved shoulder - drivers side": "paved_shoulder_driver_side",
    "Paved shoulder - passenger side": "paved_shoulder_passenger_side",
    "Street lighting": "street_lighting",
    "Vehicle parking": "vehicle_parking",
    "Service road": "service_road",
    "Roadworks": "roadworks",
    "Median type": "median_type",
    "Number of lanes": "number_of_lanes",
    "Lane width": "lane_width",
    "Curvature": "curvature",
    "Quality of curve": "quality_of_curve",
    "Grade": "grade",
    "Road condition": "road_condition",
    "Delineation": "delineation",
    "Centreline rumble strips": "centreline_rumble_strips",
    "Shoulder rumble strips": "shoulder_rumble_strips",
    "Sight distance": "sight_distance",
    "Intersection type": "intersection_type",
    "Intersection quality": "intersection_quality",
    "Intersecting road volume": "intersecting_road_volume",
    "Intersection channelisation": "intersection_channelisation",
    "Property access points": "property_access_points",
    "Vehicle flow (AADT)": "vehicle_flow",
    "Motorcycle %": "motorcycle_percent",
    "Pedestrian peak hour flow across the road": "ped_peak_hour_flow_across",
    "Pedestrian peak hour flow along the road driver-side": "ped_peak_hour_flow_along_driver_side",
    "Pedestrian peak hour flow along the road passenger-side": "ped_peak_hour_flow_along_passenger_side",
    "Bicycle peak hourly flow": "bicycle_peak_hour_flow",
    "HGV %": "hgv_percent",
    "Pedestrian channelisation ": "ped_channelisation",
    "Sidewalk - drivers side": "sidewalk_driver_side",
    "Sidewalk - passenger side": "sidewalk_passenger_side",
    "School zone warning": "school_zone_warning",
    "School zone crossing supervisor": "school_zone_crossing_supervisor",
    "Speed limit": "speed_limit",
    "Speed management": "speed_management",
    "Operating Speed (85th percentile)": "operating_speed_85th_percentile",
    "Variable speed limit": "variable_speed_limit",
    "Incident detection warning system": "incident_detection_warning_system",
    "Speed cameras": "speed_cameras",
}

def get_authenticated_session() -> tuple[requests.Session, str]:
    """
    Build a requests.Session carrying the PHPSESSID cookie, fetch the
    Demonstrator page to obtain a fresh CSRF token from the page's
    <meta name="csrf-token" content="..."> tag, and return both.
    """
    session = requests.Session()
    session.cookies.set(
        "PHPSESSID",
        SESSION_COOKIE_VALUE,
        domain="demonstrator.vida.irap.org",
    )

    page_response = session.get(DEMONSTRATOR_PAGE_URL, timeout=30)
    page_response.raise_for_status()

    match = re.search(
        r'<meta\s+name=["\']csrf-token["\']\s+content=["\']([^"\']+)["\']',
        page_response.text,
    )
    if not match:
        raise RuntimeError(
            "Could not find <meta name='csrf-token' content='...'> in the "
            "Demonstrator page HTML. Either the page URL is wrong, the "
            "session cookie is invalid/expired (check if the page text "
            "looks like a login page), or the meta tag name differs from "
            "what was found manually - re-check DevTools > View Page Source."
        )

    csrf_token = match.group(1)
    return session, csrf_token


def _as_multipart(attributes: dict) -> dict:
    """
    Convert a flat {field: value} dict into the {field: (None, str(value))}
    shape requests needs to force TRUE multipart/form-data encoding (matching
    the browser's WebKitFormBoundary payload) even though there are no actual
    files being uploaded - just text fields.
    """
    return {key: (None, str(value)) for key, value in attributes.items()}


def extract_star_ratings(result: dict) -> dict:
    """Pull the relevant star ratings (and decimal scores) out of a Demonstrator response."""
    data = result.get("data", {})

    fields = [
        "bicycle_star_rating", "car_star_rating", "motorcycle_star_rating",
        "pedestrian_star_rating", "car_star_decimal",
        "car_star", "motorcycle_star_decimal", "motorcycle_star",
        "pedestrian_star_decimal", "pedestrian_star",
        "bicycle_star_decimal", "bicycle_star"
        ]

    stars = {}
    for field in fields:
        stars[field] = data.get(field)

    return stars


def run_batch(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the Demonstrator over multiple segments (e.g. rows from your pipeline's
    GeoParquet/BigQuery output) and collect star ratings for each.

    Authenticates ONCE and reuses the session/CSRF token across all segments,
    rather than re-fetching the page per segment (faster, and less likely to
    get you rate-limited or flagged for unusual traffic).

    Select relevant columns from df, create a VIDA payload for each row,
    call the VIDA API, and append the SRS results to the original df.
    """
    session, csrf_token = get_authenticated_session()
    headers = {
        "Accept": "application/json",
        "X-CSRF-TOKEN": csrf_token,
        "X-Requested-With": "XMLHttpRequest",
    }

    # Keep only columns required by VIDA and rename them
    df = segments_df[list(COLUMN_MAPPING.keys())].rename(columns=COLUMN_MAPPING)

    # Process each row while retaining the original DataFrame index
    for index, row in df.iterrows():
        payload = row.to_dict()
        payload = {
            key: int(value) if pd.notna(value) else None
            for key, value in payload.items()
        }
        try:
            multipart_payload = _as_multipart(payload)

            validate_response = session.post(DEMONSTRATOR_VALIDATE_URL, headers=headers, files=multipart_payload, timeout=30)
            validate_response.raise_for_status()
            validate_json = validate_response.json()
            if validate_json.get("result") == "errors":
                segments_df.loc[index, "srs_error"] = f"validation failed: {validate_json.get('reports')}"
                continue

            response = session.post(DEMONSTRATOR_API_URL, headers=headers, files=multipart_payload, timeout=30)
            response.raise_for_status()
            raw = response.json()
            stars = extract_star_ratings(raw)
            
            # Append results to the same row
            for key, value in stars.items():
                segments_df.loc[index, key] = value

        except (requests.exceptions.RequestException, requests.exceptions.JSONDecodeError) as e:
            segments_df.loc[index, "srs_error"] = str(e)
    return segments_df


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df_with_srs = run_batch(df)

    output_csv = CSV_PATH.parent / f"enriched_{CSV_PATH.name}"
    df_with_srs.to_csv(output_csv, index=False)

    print(f"CSV appended with Star Ratings saved to {output_csv}")