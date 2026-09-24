"""ADB AI for Safer Roads — Street View object detection + Speed Safety Score.

Flow:
    1. Load road segments from the ADB Innovation Thailand geojson.
    2. Match 2024 accident points (speeding / cutting-in) to their segment.
    3. For each sampled segment, take its two farthest-apart accident points and
       fetch Google Street View at those coordinates.
    4. Run YOLO-World object detection and Mask2Former scene segmentation on each
       image; write labelled imagery.
    5. Compute the per-segment Speed Safety Score, and the speed / objects /
       accidents correlation bubble chart.
"""

__all__ = ["config"]
