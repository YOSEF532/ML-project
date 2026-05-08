"""
fix_candidate_labels.py
========================
Corrects candidate labels in features.csv.

Problem: The current 'label' column is a CASE-level label — every candidate
         in a positive case gets label=1, even though 95%+ are normal vessels.

Fix:     Use aneurysm_locations.csv (96×96×96 mask-space coordinates) to assign
         label=1 ONLY to candidates whose spatial location is within
         LABEL_RADIUS_MM of a known aneurysm centroid.

Coordinate scaling:
  - aneurysm_locations.csv has voxel coords in 96×96×96 prediction-mask space
  - features.csv has mm coords in preprocessed NIfTI space (0.5 mm isotropic)
  - Mapping:  nifti_mm = mask_voxel * (nifti_dim / 96) * 0.5
  - NIfTI Z-dim is computed from spatial_depth_ratio in features.csv
  - NIfTI Y/X dims are estimated from max observed spatial coords + margin
  - balanced_mra_train.csv confirms case-level aneurysm presence
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURES_CSV     = Path("preprocessed/features.csv")
ANEURYSM_CSV     = Path("aneurysm_locations.csv")
TRAIN_CSV        = Path("mra_subset_dataset/balanced_mra_train.csv")
OUTPUT_CSV       = Path("preprocessed/features.csv")   # overwrite in-place
BACKUP_CSV       = Path("preprocessed/features_backup.csv")

VOXEL_SPACING    = 0.5      # mm — isotropic after preprocessing
MASK_DIM         = 96       # prediction masks are 96×96×96
LABEL_RADIUS_MM  = 15.0     # distance threshold for positive label
DIM_MARGIN_VOX   = 10       # margin added when estimating Y/X dims


def estimate_nifti_dims(case_df: pd.DataFrame) -> tuple:
    """Estimate (Z, Y, X) voxel dimensions of the NIfTI volume for a case."""
    # Z from depth_ratio:  depth_ratio = cz / Z  →  Z = cz / depth_ratio
    valid = case_df[case_df['spatial_depth_ratio'] > 0.01]
    if len(valid) > 0:
        Z_mm_estimates = valid['spatial_z_mm'] / valid['spatial_depth_ratio']
        Z_vox = Z_mm_estimates.median() / VOXEL_SPACING
    else:
        Z_vox = case_df['spatial_z_mm'].max() / VOXEL_SPACING + DIM_MARGIN_VOX

    # Y and X from max observed spatial coords + margin
    Y_vox = case_df['spatial_y_mm'].max() / VOXEL_SPACING + DIM_MARGIN_VOX
    X_vox = case_df['spatial_x_mm'].max() / VOXEL_SPACING + DIM_MARGIN_VOX

    return float(Z_vox), float(Y_vox), float(X_vox)


def mask_to_nifti_mm(centroid_x, centroid_y, centroid_z,
                     Z_vox, Y_vox, X_vox) -> tuple:
    """
    Convert aneurysm centroid from 96×96×96 mask space to NIfTI mm space.

    Axis mapping:
      mask x (always centroid=48) → NIfTI Z (depth/slice axis)
      mask y                      → NIfTI Y
      mask z                      → NIfTI X
    """
    z_mm = centroid_x * (Z_vox / MASK_DIM) * VOXEL_SPACING
    y_mm = centroid_y * (Y_vox / MASK_DIM) * VOXEL_SPACING
    x_mm = centroid_z * (X_vox / MASK_DIM) * VOXEL_SPACING
    return z_mm, y_mm, x_mm


def main():
    print("=" * 65)
    print("FIX CANDIDATE LABELS")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    df   = pd.read_csv(FEATURES_CSV)
    anr  = pd.read_csv(ANEURYSM_CSV)
    train = pd.read_csv(TRAIN_CSV)

    print(f"Features:  {len(df)} rows, {df['case_id'].nunique()} cases")
    print(f"Aneurysm locations: {len(anr)} rows")
    print(f"Train CSV: {len(train)} cases")
    print(f"\nCurrent label distribution:\n{df['label'].value_counts().to_string()}")

    # ── Backup ────────────────────────────────────────────────────────────
    df.to_csv(BACKUP_CSV, index=False)
    print(f"\nBackup saved to {BACKUP_CSV}")

    # ── Build aneurysm lookup: UID → list of (cx, cy, cz) ────────────────
    anr_positive = anr[anr['has_aneurysm'] == 1].copy()
    anr_lookup = {}
    for _, row in anr_positive.iterrows():
        uid = str(row['SeriesInstanceUID'])
        anr_lookup.setdefault(uid, []).append(
            (float(row['centroid_x']), float(row['centroid_y']),
             float(row['centroid_z']))
        )

    # ── Build case-level label lookup from train CSV ──────────────────────
    train_label = dict(zip(
        train['SeriesInstanceUID'].astype(str),
        train['Aneurysm Present'].astype(int)
    ))

    # ── Fix labels case by case ───────────────────────────────────────────
    new_labels = np.zeros(len(df), dtype=int)
    stats = {'pos_cases': 0, 'neg_cases': 0, 'pos_cands': 0,
             'neg_cands_in_pos': 0, 'no_anr_info': 0}

    for uid, case_df in df.groupby('case_id'):
        idx = case_df.index
        uid_str = str(uid)

        # Determine if case is positive
        has_aneurysm = train_label.get(uid_str, -1)
        if has_aneurysm == -1:
            # Fallback: check aneurysm_locations
            has_aneurysm = 1 if uid_str in anr_lookup else 0

        if has_aneurysm == 0:
            # Negative case — all candidates are negative
            new_labels[idx] = 0
            stats['neg_cases'] += 1
            continue

        stats['pos_cases'] += 1

        # Positive case — check each candidate against aneurysm locations
        if uid_str not in anr_lookup:
            # Positive case but no aneurysm location info — keep as positive
            new_labels[idx] = 1
            stats['no_anr_info'] += 1
            stats['pos_cands'] += len(idx)
            continue

        # Estimate NIfTI dimensions for coordinate scaling
        Z_vox, Y_vox, X_vox = estimate_nifti_dims(case_df)

        # Convert all aneurysm centroids to NIfTI mm space
        anr_mm_list = []
        for cx, cy, cz in anr_lookup[uid_str]:
            z_mm, y_mm, x_mm = mask_to_nifti_mm(cx, cy, cz,
                                                  Z_vox, Y_vox, X_vox)
            anr_mm_list.append((z_mm, y_mm, x_mm))

        # For each candidate, compute distance to nearest aneurysm
        for i in idx:
            cand_z = df.at[i, 'spatial_z_mm']
            cand_y = df.at[i, 'spatial_y_mm']
            cand_x = df.at[i, 'spatial_x_mm']

            min_dist = float('inf')
            for az, ay, ax in anr_mm_list:
                dist = np.sqrt((cand_z - az)**2 +
                               (cand_y - ay)**2 +
                               (cand_x - ax)**2)
                min_dist = min(min_dist, dist)

            if min_dist <= LABEL_RADIUS_MM:
                new_labels[i] = 1
                stats['pos_cands'] += 1
            else:
                new_labels[i] = 0
                stats['neg_cands_in_pos'] += 1

    # ── Apply new labels ──────────────────────────────────────────────────
    df['label'] = new_labels
    df.to_csv(OUTPUT_CSV, index=False)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"RESULTS")
    print(f"{'=' * 65}")
    print(f"Positive cases:        {stats['pos_cases']}")
    print(f"Negative cases:        {stats['neg_cases']}")
    print(f"Pos cases w/o loc:     {stats['no_anr_info']}")
    print(f"Positive candidates:   {stats['pos_cands']}")
    print(f"Neg cands in pos cases:{stats['neg_cands_in_pos']}")
    print(f"\nNew label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
