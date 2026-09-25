"""
  17 Sep 2026 - Created by AW
  22 Sep 2026 - Create Linestring geometry from csv lat/lon start & end cols
  23 Sep 2026 - Extract pKey from "Image reference" for Mapillary iframe embed
  23 Sep 2026 - Add "<column> text" human-readable labels for every coded
                iRAP column (dashboard-only -- NOT written back to the CSV,
                that stage stays numeric-only per the upload spec). Code
                tables are the iRAP v3.10 Coding Manual's own "Quick Coding
                and Validation Guide" (pages 107-115), not guessed.

  Note: geopandas's pyproj failed to pip install on MacOS python 3.14,
  so need to use pandas + shapely + json.
"""

import pandas as pd
import numpy as np
import json
from shapely import LineString
from shapely.geometry import mapping

IN_CSV    = "../output/main/SRS_rating(ViDA_API).csv"
OUT_FILE  = "../output/main/SRS_rating(ViDA_API).geojson"

CODE_LABELS = {
    "Carriageway label": {
        1: "Carriageway A", 2: "Carriageway B", 3: "Undivided road",
        4: "Motorcycle facility A", 5: "Motorcycle facility B",
    },
    "Roadworks": {1: "No roadworks", 2: "Minor roadworks", 3: "Major roadworks"},
    "Upgrade cost": {1: "Low", 2: "Medium", 3: "High"},
    "Incident detection warning system": {1: "Not present", 2: "Present"},
    "Motorcycle %": {
        2: "0%", 3: "1 to 5%", 4: "6 to 10%", 5: "11 to 20%", 6: "21 to 40%",
        7: "41 to 60%", 8: "61 to 80%", 9: "81 to 99%", 10: "100%",
    },
    "HGV %": {
        2: "0%", 3: "1 to 5%", 4: "6 to 10%", 5: "11 to 15%", 6: "16 to 20%",
        7: "21 to 30%", 8: "31 to 40%", 9: "≥40%",
    },
    "Pedestrian peak hour flow across the road": {
        1: "0", 2: "1 to 5", 3: "6 to 25", 4: "26 to 50", 5: "51 to 100",
        6: "101 to 200", 7: "201 to 300", 8: "301 to 400", 9: "401 to 500",
        10: "501 to 900", 11: "900+",
    },
    "Variable speed limit": {1: "Not present", 2: "Present"},
    "Speed differential": {1: "Not present", 2: "Present"},
    "Speed management": {1: "Not present", 2: "Present"},
    "Speed cameras": {1: "No speed camera", 2: "Fixed speed camera", 3: "Average speed camera"},
    "Number of lanes": {
        1: "One lane", 2: "Two lanes", 3: "Three lanes", 4: "Four or more lanes",
        5: "Two and one lanes", 6: "Three and two lanes",
    },
    "Lane width": {1: "Wide ≥3.25m", 2: "Medium 2.75 to <3.25m", 3: "Narrow <2.75m"},
    "Curvature": {1: "Straight or gently curving", 2: "Moderate", 3: "Sharp", 4: "Very sharp"},
    "Quality of curve": {1: "Adequate", 2: "Poor", 3: "Not applicable"},
    "Median type": {
        1: "Safety barrier – metal", 2: "Safety barrier – concrete",
        3: "Physical median ≥20m wide", 4: "Physical median 10 to <20m wide",
        5: "Physical median 5 to <10m wide", 6: "Physical median 1 to <5m wide",
        7: "Physical median 0 to <1m wide", 8: "Continuous central turning lane",
        9: "Flexible posts", 10: "Continuous wide median markings ≥0.6m",
        11: "Centreline", 12: "Safety barrier – motorcycle friendly",
        13: "One way", 14: "Double centreline",
        15: "Safety barrier – wire rope", 16: "Broken wide median markings ≥0.6m",
    },
    "Skid resistance": {
        1: "Sealed – adequate", 2: "Sealed – medium", 3: "Sealed – poor",
        4: "Unsealed – adequate", 5: "Unsealed – poor",
    },
    "Road condition": {1: "Good", 2: "Medium", 3: "Poor"},
    "Vehicle parking": {1: "None", 2: "One side", 3: "Two side"},
    "Grade": {1: "<7.5%", 4: "7.5 to <10%", 5: "≥10%"},
    "Sight distance": {1: "Adequate", 2: "Poor"},
    "Delineation": {1: "Adequate", 2: "Poor"},
    "Street lighting": {1: "Not present", 2: "Present"},
    "Service road": {1: "Not present", 2: "Present"},
    "Centreline rumble strips": {1: "Not present", 2: "Present"},
    "Shoulder rumble strips": {1: "Not present", 2: "Present"},
    "Roadside severity - drivers side distance": {1: "0 to <1m", 2: "1 to <5m", 3: "5 to <10m", 4: "≥10m"},
    "Roadside severity - passenger side distance": {1: "0 to <1m", 2: "1 to <5m", 3: "5 to <10m", 4: "≥10m"},
    "Roadside severity - drivers side object": {
        1: "Safety barrier – metal", 2: "Safety barrier – concrete",
        3: "Safety barrier – motorcycle friendly", 4: "Safety barrier – wire rope",
        5: "Aggressive vertical face", 6: "Upwards slope – roll over",
        7: "Upwards slope – no roll over", 8: "Deep drainage ditch",
        9: "Downwards slope", 10: "Cliff", 11: "Tree ≥10cm",
        12: "Rigid sign, post or pole ≥10cm", 13: "Rigid structure or building",
        14: "Semi-rigid structure or building", 15: "Unsafe barrier end",
        16: "Low rigid object ≥20cm high", 17: "No object",
    },
    "Roadside severity - passenger side object": {
        1: "Safety barrier – metal", 2: "Safety barrier – concrete",
        3: "Safety barrier – motorcycle friendly", 4: "Safety barrier – wire rope",
        5: "Aggressive vertical face", 6: "Upwards slope – roll over",
        7: "Upwards slope – no roll over", 8: "Deep drainage ditch",
        9: "Downwards slope", 10: "Cliff", 11: "Tree ≥10cm",
        12: "Rigid sign, post or pole ≥10cm", 13: "Rigid structure or building",
        14: "Semi-rigid structure or building", 15: "Unprotected safety barrier end",
        16: "Low rigid object ≥20cm high", 17: "No object",
    },
    "Paved shoulder - drivers side": {1: "Wide ≥2.4m", 2: "Medium 1 to <2.4m", 3: "Narrow 0 to <1m", 4: "None"},
    "Paved shoulder - passenger side": {1: "Wide ≥2.4m", 2: "Medium 1 to <2.4m", 3: "Narrow 0 to <1m", 4: "None"},
    "Intersection type": {
        1: "Merge lane", 2: "Roundabout", 3: "3 leg with turn lane", 4: "3 leg",
        5: "3 leg signalised with turn lane", 6: "3 leg signalised",
        7: "4+ leg with turn lane", 8: "4+ leg", 9: "4+ leg signalised with turn lane",
        10: "4+ leg signalised", 12: "None", 13: "Railway crossing – passive",
        14: "Railway crossing – active", 15: "Median crossing point",
        16: "Median crossing point with turn lane", 17: "Mini roundabout",
        18: "4 leg staggered with turn lane", 19: "4 leg staggered",
        20: "3 leg without median gap", 22: "Short merge/weaving lane", 23: "Diverge lane",
    },
    "Intersection quality": {1: "Adequate", 2: "Poor", 3: "Not applicable"},
    "Intersection channelisation": {1: "Not present", 2: "Present"},
    "Property access points": {
        1: "Commercial access 1+", 2: "Residential access 3+",
        3: "Residential access 1 or 2", 4: "None",
    },
    "Land use - drivers side": {
        1: "Undeveloped areas", 2: "Farming and agricultural", 3: "Residential",
        4: "Commercial", 5: "Not Recorded", 6: "Educational", 7: "Industrial and manufacturing",
    },
    "Land use - passenger side": {
        1: "Undeveloped areas", 2: "Farming and agricultural", 3: "Residential",
        4: "Commercial", 5: "Not Recorded", 6: "Educational", 7: "Industrial and manufacturing",
    },
    "Area type": {1: "Rural", 2: "Urban"},
    "Crossing facility - inspected road": {
        1: "Grade separated facility", 2: "Signalised crossing with refuge",
        3: "Signalised crossing", 4: "Marked crossing with refuge", 5: "Marked crossing",
        6: "Unmarked crossing with refuge", 7: "No facility", 8: "Unmarked crossing",
        14: "Raised marked crossing with refuge", 15: "Raised marked crossing",
        16: "Raised unmarked crossing with refuge", 17: "Raised unmarked crossing",
    },
    "Crossing facility - side road": {
        1: "Grade separated facility", 2: "Signalised crossing with refuge",
        3: "Signalised crossing", 4: "Marked crossing with refuge", 5: "Marked crossing",
        6: "Unmarked crossing with refuge", 7: "No facility", 8: "Unmarked crossing",
        14: "Raised marked crossing with refuge", 15: "Raised marked crossing",
        16: "Raised unmarked crossing with refuge", 17: "Raised unmarked crossing",
    },
    "Crossing quality": {1: "Adequate", 2: "Poor", 3: "Not applicable"},
    "Pedestrian channelisation ": {1: "Not present", 2: "Present"},
    "Sidewalk - drivers side": {
        1: "Sidewalk with barrier", 2: "Sidewalk ≥3m from road",
        3: "Sidewalk 1 to <3m from road", 4: "Sidewalk 0 to <1m from road",
        5: "None", 6: "Medium quality sidewalk", 7: "Poor quality sidewalk",
        8: "Shared use path",
    },
    "Sidewalk - passenger side": {
        1: "Sidewalk with barrier", 2: "Sidewalk ≥3m from road",
        3: "Sidewalk 1 to <3m from road", 4: "Sidewalk 0 to <1m from road",
        5: "None", 6: "Medium quality sidewalk", 7: "Poor quality sidewalk",
        8: "Shared use path",
    },
    "PTW facilities": {
        1: "Motorcycle path – one way with barrier", 2: "Motorcycle path – one way",
        3: "Motorcycle path – two way with barrier", 4: "Motorcycle path – two way",
        5: "Motorcycle lane on roadway", 6: "None",
    },
    "Cycling facilities": {
        1: "Segregated bicycle path with barrier", 2: "Segregated bicycle path",
        3: "Bicycle lane", 4: "None", 5: "Extra wide outside ≥4.2m",
        6: "Signed shared roadway", 7: "Shared use path",
    },
    "School zone warning": {
        1: "School zone – flashing beacons", 2: "School zone – static signs or road markings",
        3: "No school zone warning (school present)", 4: "Not applicable",
    },
    "School zone crossing supervisor": {1: "Present", 2: "Not present", 3: "Not applicable"},
}
# Pedestrian/bicycle flow columns all share the same code scale as the
# "across the road" one above.
_PED_BIKE_FLOW = CODE_LABELS["Pedestrian peak hour flow across the road"]
CODE_LABELS["Pedestrian peak hour flow along the road driver-side"] = _PED_BIKE_FLOW
CODE_LABELS["Pedestrian peak hour flow along the road passenger-side"] = _PED_BIKE_FLOW
CODE_LABELS["Bicycle peak hourly flow"] = _PED_BIKE_FLOW


# pandas' default na_values list, minus "NA" -- ViDA's API returns the literal
# string "NA" to mean "Not Applicable" (e.g. bicycle_star when a segment has
# zero recorded cyclist flow -- a deliberate, meaningful answer, not missing
# data). Reading with the default na_values silently turns that into NaN,
# indistinguishable from a real data gap. Every other standard missing-value
# token is still recognised as NaN.
_NA_VALUES_EXCEPT_NA = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "None", "n/a", "nan", "null",
]


def _code_text(column: str, value) -> str | None:
    labels = CODE_LABELS.get(column)
    if labels is None or value is None:
        return None
    try:
        return labels.get(int(value))
    except (TypeError, ValueError):
        return None


def convert(in_csv: str, out_file: str) -> None:
    """Read in_csv (needs Latitude/Longitude start/end columns), write
    out_file as a GeoJSON FeatureCollection of LineString features -- one
    per row, every CSV column carried over as a property, plus the pKey /
    "<column> text" additions above. Reusable so other one-off CSVs (e.g. a
    speed-sensitivity scenario batch) can go through the same conversion
    without duplicating the CODE_LABELS table."""
    df = pd.read_csv(in_csv, keep_default_na=False, na_values=_NA_VALUES_EXCEPT_NA)

    # Replace NaN with None across the entire DataFrame
    df = df.fillna(np.nan).replace([np.nan], [None])

    features = []

    for _, row in df.iterrows():
        # Create Linestring with Shapely geometry
        geometry = LineString([
                (row["Longitude start"], row["Latitude start"]),
                (row["Longitude end"], row["Latitude end"])
            ])
        # Print the WKT representation
        #print(geometry.wkt)

        # Use all columns as GeoJSON properties
        properties = row.to_dict()

        # Extract pKey from "Image reference" for Mapillary iframe embed
        text = row["Image reference"]
        if "pKey=" in text:
            result = text.split("pKey=", 1)[1]
            # If you want to stop at the next '&' or end of string:
            result = result.split("&")[0]
            #print(result)
            properties["pKey"] = int(result)

        # Add "<column> text" human-readable labels for every coded column
        # (dashboard-only -- see CODE_LABELS above, not written back to the CSV)
        for column, value in row.items():
            label = _code_text(column, value)
            if label is not None:
                properties[f"{column.strip()} text"] = label

        feature = {
            "type": "Feature",
            "geometry": mapping(geometry),
            "properties": properties
        }

        features.append(feature)

    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # Write GeoJSON file
    # ensure_ascii=False so symbols like the "≥"/"–" text labels above
    # (>=, en dash, etc.) are written as real UTF-8 characters, not \uXXXX
    # escapes -- still valid JSON either way, but keeps the raw file readable.
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    print(f"Created {out_file}")


if __name__ == "__main__":
    convert(IN_CSV, OUT_FILE)
