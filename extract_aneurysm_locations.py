"""
Extract aneurysm locations from predicted mask files in results.zip.
Each mask is a 96x96x96 float16 probability volume.
We threshold at 0.5, find connected components, and export per-region
centroid, bounding box, volume (voxel count), and max probability to CSV.
"""

import zipfile
import numpy as np
import io
import csv
from scipy import ndimage

ZIP_PATH = "results.zip"
OUTPUT_CSV = "aneurysm_locations.csv"
THRESHOLD = 0.5

def extract_series_uid(filename: str) -> str:
    """Extract the SeriesInstanceUID from the mask filename."""
    # e.g. "filtered_masks/1.2.826.0...647_mask.npz" -> "1.2.826.0...647"
    basename = filename.split("/")[-1]          # drop folder
    uid = basename.replace("_mask.npz", "")     # drop suffix
    return uid

def process_mask(arr: np.ndarray, threshold: float = THRESHOLD):
    """Threshold the probability mask and extract per-region stats."""
    binary = (arr > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []  # no aneurysm detected

    labeled, num_features = ndimage.label(binary)
    regions = []
    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        coords = np.argwhere(region_mask)
        centroid = coords.mean(axis=0)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        voxel_count = len(coords)
        max_prob = float(arr[region_mask].max())

        regions.append({
            "centroid_x": round(float(centroid[0]), 2),
            "centroid_y": round(float(centroid[1]), 2),
            "centroid_z": round(float(centroid[2]), 2),
            "bbox_min_x": int(bbox_min[0]),
            "bbox_min_y": int(bbox_min[1]),
            "bbox_min_z": int(bbox_min[2]),
            "bbox_max_x": int(bbox_max[0]),
            "bbox_max_y": int(bbox_max[1]),
            "bbox_max_z": int(bbox_max[2]),
            "voxel_count": voxel_count,
            "max_probability": round(max_prob, 4),
        })
    return regions

def main():
    z = zipfile.ZipFile(ZIP_PATH)
    mask_files = sorted([n for n in z.namelist() if n.endswith(".npz")])

    rows = []
    positive = 0
    negative = 0

    for i, name in enumerate(mask_files):
        uid = extract_series_uid(name)
        data = np.load(io.BytesIO(z.read(name)))
        arr = data["mask"]
        regions = process_mask(arr)

        if not regions:
            negative += 1
            # Still record the series with label=0 and no location
            rows.append({
                "SeriesInstanceUID": uid,
                "has_aneurysm": 0,
                "region_id": 0,
                "centroid_x": "",
                "centroid_y": "",
                "centroid_z": "",
                "bbox_min_x": "",
                "bbox_min_y": "",
                "bbox_min_z": "",
                "bbox_max_x": "",
                "bbox_max_y": "",
                "bbox_max_z": "",
                "voxel_count": 0,
                "max_probability": 0,
            })
        else:
            positive += 1
            for rid, region in enumerate(regions, start=1):
                rows.append({
                    "SeriesInstanceUID": uid,
                    "has_aneurysm": 1,
                    "region_id": rid,
                    **region,
                })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(mask_files)} masks...")

    # Write CSV
    fieldnames = [
        "SeriesInstanceUID", "has_aneurysm", "region_id",
        "centroid_x", "centroid_y", "centroid_z",
        "bbox_min_x", "bbox_min_y", "bbox_min_z",
        "bbox_max_x", "bbox_max_y", "bbox_max_z",
        "voxel_count", "max_probability",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print(f"  Positive (with aneurysm): {positive}")
    print(f"  Negative (no aneurysm):   {negative}")
    print(f"  Total masks processed:    {len(mask_files)}")

if __name__ == "__main__":
    main()
