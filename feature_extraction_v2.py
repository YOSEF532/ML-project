"""
feature_extraction_v2.py — Feature extraction with candidate-level label assignment.
======================================================================================
Combines:
  • feature_extraction.py  — Frangi vesselness map → ~55 features per candidate
  • Notebook Cells 1-5     — anisotropic heatmap mask + localizer coordinate handling
  • balanced_mra_train.csv — case-level and artery-location labels
  • train_localizers.csv   — (x, y) aneurysm locations in original 512-px space

Key improvements over v1
-------------------------
    1. For NEGATIVE cases (Aneurysm Present=0):
         all candidates → cand_label = 0.

    2. For POSITIVE cases WITH localizer entries:
         a candidate is positive (cand_label=1) if ANY known aneurysm x,y
         falls within LABEL_RADIUS_MM of the candidate centroid (XY plane only,
         because loc_df has no z coordinate).
         Remaining candidates → cand_label = 0.

    3. For POSITIVE cases WITHOUT any localizer entry:
         candidates → cand_label = -1  (ambiguous; exclude from training or
         treat as weak positives; see --keep_ambiguous flag).

  Coordinate scaling:
    loc_df coords are in original 512×512 pixel space.
    NIfTI volumes are at 0.5 mm isotropic after resampling.
    Conversion: loc_vox = loc_px * (nii_axis_size / 512)
    Distance check is in mm (XY only); Z uncertainty is acknowledged.

  NEW — Extra columns
  -------------------
    • loc_*  : the 13 artery-location binary flags (case-level, from CSV)
    • patient_age, patient_sex : demographics (from balanced_mra_train.csv)
    • xy_dist_to_nearest_loc_mm : mm distance to the nearest localizer point
                                   (useful as a ranking feature; 0 if no locs)

Usage
-----
    python feature_extraction_v2.py \\
        --preprocessed_dir  path/to/preprocessed \\
        --train_csv         balanced_mra_train.csv \\
        --localizer_csv     train_localizers.csv \\
        --output            features_v2.csv

    # Discard ambiguous candidates (positive case, no localizer entry):
    python feature_extraction_v2.py ... --drop_ambiguous

    # Treat ambiguous as positive (conservative recall-first strategy):
    python feature_extraction_v2.py ... --ambiguous_as_positive

    # Debug on 20 cases:
    python feature_extraction_v2.py ... --limit 20
"""

import argparse
import ast
import warnings
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage import measure, morphology
from skimage.feature import blob_log

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Fixed constants matching preprocess_worker.py ─────────────────────────────
VOXEL_SPACING = (0.5, 0.5, 0.5)    # mm — isotropic after resampling
VOXEL_VOL_MM3 = 0.5 ** 3           # 0.125 mm³
ORIG_IMG_SIZE = 512                  # original DICOM pixel grid (square assumed)

# ── Vesselness candidate-generation parameters ────────────────────────────────
VESSEL_THRESH   = 0.15
MIN_VOXELS      = 8
MAX_VOXELS      = 50_000

BLOB_MIN_SIGMA  = 2.0
BLOB_MAX_SIGMA  = 20.0
BLOB_NUM_SIGMA  = 12
BLOB_THRESHOLD  = 0.04
BLOB_OVERLAP    = 0.3
DEDUP_DIST_MM   = 4.0

# ── Candidate-label assignment ────────────────────────────────────────────────
LABEL_RADIUS_MM = 10.0   # XY distance threshold for positive label assignment

# ── Location columns (must match balanced_mra_train.csv) ─────────────────────
LOCATION_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
]


# =============================================================================
# 1.  SCAN DISCOVERY  (updated: reads balanced_mra_train.csv + localizer csv)
# =============================================================================

def discover_cases(preprocessed_dir: Path, train_csv: Path, localizer_csv: Path):
    """
    Returns list of dicts:
      { uid, nii_path, has_aneurysm, location_flags, patient_age, patient_sex,
        loc_xy_voxel }

    loc_xy_voxel : list of (y_vox, x_vox) tuples in the NIfTI voxel space.
                   Empty list if no localizer entries for this case.
                   Conversion:  px * (nii_shape_axis / ORIG_IMG_SIZE)
                   (shape is read later at load time, so we store raw px here
                    and scale at feature-extraction time.)
    """
    train_df = pd.read_csv(train_csv)
    loc_df   = pd.read_csv(localizer_csv)

    # Normalise column names
    train_df.columns = train_df.columns.str.strip()
    loc_df.columns   = loc_df.columns.str.strip()

    # Build localizer lookup: UID → list of (x_px, y_px)
    loc_lookup: dict[str, list[tuple[float, float]]] = {}
    for _, row in loc_df.iterrows():
        uid = str(row['SeriesInstanceUID'])
        try:
            coords = ast.literal_eval(row['coordinates'])
            x_px   = float(coords['x'])
            y_px   = float(coords['y'])
        except Exception:
            continue
        loc_lookup.setdefault(uid, []).append((x_px, y_px))

    log.info(f"Train CSV: {len(train_df)} cases  |  "
             f"Localizer entries: {len(loc_df)} rows for "
             f"{len(loc_lookup)} unique UIDs")

    cases = []
    for _, row in train_df.iterrows():
        uid      = str(row['SeriesInstanceUID'])
        nii_path = preprocessed_dir / uid / f"{uid}_final.nii.gz"
        if not nii_path.exists():
            log.debug(f"  [MISSING NII] {uid}")
            continue

        # 13 artery location binary flags
        loc_flags = {
            col: int(row[col]) if col in row.index else 0
            for col in LOCATION_COLS
        }

        cases.append({
            'uid':          uid,
            'nii_path':     nii_path,
            'has_aneurysm': int(row['Aneurysm Present']),
            'location_flags': loc_flags,
            'patient_age':  int(row['PatientAge'])   if 'PatientAge'  in row.index else -1,
            'patient_sex':  str(row['PatientSex'])   if 'PatientSex'  in row.index else 'Unknown',
            'loc_xy_px':    loc_lookup.get(uid, []),  # raw px coords, scaled at extract time
        })

    log.info(f"Cases with NIfTI files: {len(cases)}")
    pos = sum(c['has_aneurysm'] for c in cases)
    log.info(f"  Positive: {pos}  |  Negative: {len(cases) - pos}")
    n_with_locs = sum(1 for c in cases if c['loc_xy_px'])
    log.info(f"  Positive with localizer coords: {n_with_locs}")
    return cases


# =============================================================================
# 2.  CANDIDATE-LEVEL LABEL ASSIGNMENT
# =============================================================================

def assign_candidate_labels(
    candidates:        list[dict],
    has_aneurysm:      int,
    loc_xy_px:         list[tuple[float, float]],
    nii_shape:         tuple[int, int, int],   # (Z, Y, X)
    label_radius_mm:   float = LABEL_RADIUS_MM,
    ambiguous_as_pos:  bool  = False,
) -> list[dict]:
    """
    Assigns cand_label and xy_dist_to_nearest_loc_mm to every candidate.

    cand_label values:
      0   — confirmed negative (vessel segment, not an aneurysm)
      1   — confirmed positive  (within label_radius_mm of a known aneurysm)
     -1   — ambiguous (positive case but no localizer coords available)

    Coordinate conversion:
      The loc_df x,y coordinates are in the original 512×512 pixel space.
      The NIfTI volume is (Z, Y, X) at 0.5 mm isotropic.
      We assume the X axis of the NIfTI volume corresponds to the x axis of
      the original image (and similarly for Y), so:
          x_vox = x_px * (nii_X / ORIG_IMG_SIZE)
          y_vox = y_px * (nii_Y / ORIG_IMG_SIZE)
      Distance is computed in mm (× VOXEL_SPACING) using XY only.
    """
    _, nii_Y, nii_X = nii_shape
    scale_x = nii_X / ORIG_IMG_SIZE
    scale_y = nii_Y / ORIG_IMG_SIZE

    # Convert localizer coords to voxel space (XY only)
    loc_vox = [
        (y_px * scale_y, x_px * scale_x)
        for (x_px, y_px) in loc_xy_px
    ]

    labeled = []
    for cand in candidates:
        cz, cy, cx = cand['centroid']
        cand = dict(cand)  # shallow copy — don't mutate original

        if has_aneurysm == 0:
            # Negative case — all candidates are negatives
            cand['cand_label']                 = 0
            cand['xy_dist_to_nearest_loc_mm']  = float('nan')

        elif not loc_vox:
            # Positive case, no localizer coords — ambiguous
            cand['cand_label']                 = 1 if ambiguous_as_pos else -1
            cand['xy_dist_to_nearest_loc_mm']  = float('nan')

        else:
            # Positive case with localizer coords — spatial matching
            # Compute XY distance in mm to every localizer point
            dists_mm = [
                np.sqrt(
                    ((cy - ly_vox) * VOXEL_SPACING[1]) ** 2 +
                    ((cx - lx_vox) * VOXEL_SPACING[2]) ** 2
                )
                for (ly_vox, lx_vox) in loc_vox
            ]
            nearest_mm = float(min(dists_mm))
            cand['xy_dist_to_nearest_loc_mm'] = nearest_mm
            cand['cand_label'] = 1 if nearest_mm <= label_radius_mm else 0

        labeled.append(cand)
    return labeled


# =============================================================================
# 3.  CANDIDATE GENERATION  (with BUG FIX 1 applied)
# =============================================================================

def generate_candidates(vessel_map: np.ndarray):
    """
    Returns (label_map, candidates).

    BUG FIX 1 applied: LoG blobs now only claim voxels that are background
    (label_map == 0), preventing overwrite of existing CC candidates which
    previously caused 0-voxel ValueError in extract_features.
    """
    spacing = np.array(VOXEL_SPACING, dtype=float)

    # ── A: Threshold + Connected Components ──────────────────────────────────
    binary    = vessel_map > VESSEL_THRESH
    binary    = morphology.remove_small_objects(binary, min_size=MIN_VOXELS)
    label_map, n_comp = ndimage.label(binary)
    label_map = label_map.astype(np.int32)
    log.debug(f"  CC components raw: {n_comp}")

    props = measure.regionprops(label_map, intensity_image=vessel_map)
    candidates = []
    for prop in props:
        size = prop.area
        if not (MIN_VOXELS <= size <= MAX_VOXELS):
            continue
        candidates.append({
            'label_id':        prop.label,
            'centroid':        prop.centroid,
            'bbox':            prop.bbox,
            'size_voxels':     size,
            'method':          'threshold_cc',
            'peak_vesselness': float(prop.max_intensity),
            'mean_vesselness': float(prop.mean_intensity),
        })
    log.debug(f"  After size filter (CC): {len(candidates)}")

    # ── B: LoG Blob Detection on axial MIP ───────────────────────────────────
    mip = np.max(vessel_map, axis=0)
    try:
        blobs = blob_log(
            mip,
            min_sigma=BLOB_MIN_SIGMA,
            max_sigma=BLOB_MAX_SIGMA,
            num_sigma=BLOB_NUM_SIGMA,
            threshold=BLOB_THRESHOLD,
            overlap=BLOB_OVERLAP,
        )
        log.debug(f"  LoG blobs found: {len(blobs)}")
        max_existing = int(label_map.max())

        for b_idx, blob in enumerate(blobs):
            y, x, sigma = blob
            r   = max(1, int(sigma * np.sqrt(2)))
            y_c = int(np.clip(y, 0, vessel_map.shape[1] - 1))
            x_c = int(np.clip(x, 0, vessel_map.shape[2] - 1))
            z_c = int(np.argmax(vessel_map[:, y_c, x_c]))

            z0 = max(0, z_c - r);  z1 = min(vessel_map.shape[0], z_c + r + 1)
            y0 = max(0, y_c - r);  y1 = min(vessel_map.shape[1], y_c + r + 1)
            x0 = max(0, x_c - r);  x1 = min(vessel_map.shape[2], x_c + r + 1)

            cz_, cy_, cx_ = np.ogrid[:z1-z0, :y1-y0, :x1-x0]
            sphere = (
                (cz_ - (z_c - z0)) ** 2 +
                (cy_ - (y_c - y0)) ** 2 +
                (cx_ - (x_c - x0)) ** 2
            ) <= r ** 2

            # ── BUG FIX 1: only claim background voxels ───────────────────
            sub_map = label_map[z0:z1, y0:y1, x0:x1]
            free    = sphere & (sub_map == 0)   # ← was: sphere (overwrote CC labels)
            if free.sum() < MIN_VOXELS:
                continue

            new_id           = max_existing + b_idx + 1
            sub_map[free]    = new_id
            label_map[z0:z1, y0:y1, x0:x1] = sub_map
            roi_vals         = vessel_map[z0:z1, y0:y1, x0:x1][free]  # ← use free mask

            candidates.append({
                'label_id':        new_id,
                'centroid':        (float(z_c), float(y_c), float(x_c)),
                'bbox':            (z0, y0, x0, z1, y1, x1),
                'size_voxels':     int(free.sum()),
                'method':          'blob_log',
                'peak_vesselness': float(roi_vals.max())  if roi_vals.size else 0.0,
                'mean_vesselness': float(roi_vals.mean()) if roi_vals.size else 0.0,
            })
    except Exception as e:
        log.warning(f"  LoG blob detection failed: {e}")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    candidates = _deduplicate(candidates, dist_thresh_mm=DEDUP_DIST_MM, spacing=spacing)
    return label_map, candidates


def _deduplicate(candidates, dist_thresh_mm, spacing):
    if not candidates:
        return candidates
    c_arr = np.array([c['centroid'] for c in candidates]) * spacing
    keep  = [True] * len(candidates)
    for i in range(len(candidates)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(candidates)):
            if not keep[j]:
                continue
            if np.linalg.norm(c_arr[i] - c_arr[j]) < dist_thresh_mm:
                if candidates[i]['peak_vesselness'] >= candidates[j]['peak_vesselness']:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [c for c, k in zip(candidates, keep) if k]


# =============================================================================
# 4.  FEATURE EXTRACTION  (with BUG FIXes 2-6 applied)
# =============================================================================

def extract_features(candidate: dict, vessel_map: np.ndarray, label_map: np.ndarray) -> dict:
    """
    Extracts ~55 features per candidate.

    BUG FIX 2: shape_sphericity clamped to [0, 1].
    BUG FIX 3: v_energy is mean-squared (size-normalised).
    BUG FIX 4: spatial_depth_ratio documented correctly.
    BUG FIX 5: meta_method_blob_log encoded as 0/1 int.
    BUG FIX 6: v_entropy guarded against near-zero voxel sums.
    """
    lid  = candidate['label_id']
    bbox = candidate['bbox']
    z0, y0, x0, z1, y1, x1 = bbox
    eps  = 1e-9
    feats = {}

    voxels     = vessel_map[label_map == lid]
    local_mask = (label_map[z0:z1, y0:y1, x0:x1] == lid)
    n_vox      = len(voxels)
    volume_mm3 = n_vox * VOXEL_VOL_MM3

    if n_vox == 0:
        raise ValueError(f'Candidate {lid} has 0 voxels — likely overwritten (should not '
                         f'happen after BUG FIX 1)')

    # ── A: Vesselness Statistics ─────────────────────────────────────────────
    feats['v_mean']     = float(np.mean(voxels))
    feats['v_std']      = float(np.std(voxels))
    feats['v_min']      = float(np.min(voxels))
    feats['v_max']      = float(np.max(voxels))
    feats['v_range']    = feats['v_max'] - feats['v_min']
    feats['v_median']   = float(np.median(voxels))
    feats['v_p10']      = float(np.percentile(voxels, 10))
    feats['v_p25']      = float(np.percentile(voxels, 25))
    feats['v_p75']      = float(np.percentile(voxels, 75))
    feats['v_p90']      = float(np.percentile(voxels, 90))
    feats['v_iqr']      = feats['v_p75'] - feats['v_p25']
    feats['v_skewness'] = float(skew(voxels))     if n_vox > 2 else 0.0
    feats['v_kurtosis'] = float(kurtosis(voxels)) if n_vox > 2 else 0.0

    # BUG FIX 3: normalised by voxel count (mean-squared, not sum-squared)
    feats['v_energy']   = float(np.mean(voxels ** 2))

    # BUG FIX 6: guard against near-zero voxel sum
    total = voxels.sum()
    if total > eps:
        p = voxels / total
        feats['v_entropy'] = float(-np.sum(p * np.log2(p + eps)))
    else:
        feats['v_entropy'] = 0.0

    feats['v_coeff_var'] = feats['v_std'] / (feats['v_mean'] + eps)

    # ── B: Shape / Morphology ────────────────────────────────────────────────
    feats['shape_volume_vox']  = float(n_vox)
    feats['shape_volume_mm3']  = volume_mm3

    sa = 0.0
    if n_vox >= 10:
        try:
            verts, faces, _, _ = measure.marching_cubes(
                local_mask.astype(float), level=0.5,
                spacing=VOXEL_SPACING)
            sa = float(measure.mesh_surface_area(verts, faces))
        except Exception:
            sa = 0.0
    feats['shape_surface_area_mm2'] = sa

    # BUG FIX 2: clamp sphericity to [0, 1]
    raw_sphericity = (
        (np.pi ** (1 / 3)) * ((6 * volume_mm3) ** (2 / 3)) / sa
        if sa > 0 else 0.0
    )
    feats['shape_sphericity'] = float(np.clip(raw_sphericity, 0.0, 1.0))

    dz   = (z1 - z0) * VOXEL_SPACING[0]
    dy   = (y1 - y0) * VOXEL_SPACING[1]
    dx   = (x1 - x0) * VOXEL_SPACING[2]
    dims = sorted([dz, dy, dx])

    feats['shape_bbox_dz_mm']    = dz
    feats['shape_bbox_dy_mm']    = dy
    feats['shape_bbox_dx_mm']    = dx
    feats['shape_elongation']    = dims[2] / (dims[0] + eps)
    feats['shape_flatness']      = dims[0] / (dims[1] + eps)
    feats['shape_compactness']   = volume_mm3 / (dims[2] ** 3 + eps)
    feats['shape_equiv_diam_mm'] = float(2 * ((3 * volume_mm3) / (4 * np.pi)) ** (1 / 3))

    # ── C: Vesselness Texture ────────────────────────────────────────────────
    roi = vessel_map[z0:z1, y0:y1, x0:x1]
    gz  = ndimage.sobel(roi, axis=0)
    gy  = ndimage.sobel(roi, axis=1)
    gx  = ndimage.sobel(roi, axis=2)
    grad_mag  = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
    grad_vals = grad_mag[local_mask]

    feats['tex_grad_mean'] = float(np.mean(grad_vals)) if grad_vals.size else 0.0
    feats['tex_grad_std']  = float(np.std(grad_vals))  if grad_vals.size else 0.0
    feats['tex_grad_max']  = float(np.max(grad_vals))  if grad_vals.size else 0.0

    ball_r = min(3, max(1, min(local_mask.shape) // 2))
    try:
        dilated = ndimage.binary_dilation(local_mask, structure=morphology.ball(ball_r))
        shell   = roi[dilated & ~local_mask]
    except Exception:
        shell = np.array([], dtype=np.float32)

    feats['tex_shell_mean']     = float(np.mean(shell)) if shell.size > 0 else 0.0
    feats['tex_shell_std']      = float(np.std(shell))  if shell.size > 0 else 0.0
    feats['tex_local_contrast'] = float(feats['v_mean'] / (feats['tex_shell_mean'] + eps))
    feats['tex_peak_to_shell']  = float(feats['v_max']  / (feats['tex_shell_mean'] + eps))

    z_c   = (z1 - z0) // 2
    sl_v  = roi[z_c][local_mask[z_c]]
    if sl_v.size > 1:
        q      = np.digitize(sl_v, np.percentile(sl_v, np.linspace(0, 100, 9))) - 1
        glcm_e = float(np.sum((np.bincount(q, minlength=8) / len(q)) ** 2))
    else:
        glcm_e = 0.0
    feats['tex_glcm_proxy_energy'] = glcm_e

    # ── D: Blob Geometry (regionprops) ───────────────────────────────────────
    local_label = local_mask.astype(np.int32)
    rp_list = measure.regionprops(local_label, intensity_image=roi)
    if rp_list:
        rp = rp_list[0]
        feats['shape_solidity'] = float(rp.solidity)
        feats['shape_extent']   = float(rp.extent)
        if hasattr(rp, 'axis_major_length') and hasattr(rp, 'axis_minor_length'):
            ax_maj = float(rp.axis_major_length) * VOXEL_SPACING[0]
            ax_min = float(rp.axis_minor_length) * VOXEL_SPACING[0]
        else:
            ax_maj = float(max(dz, dy, dx))
            ax_min = float(min(dz, dy, dx))
        feats['shape_axis_major_mm'] = ax_maj
        feats['shape_axis_minor_mm'] = ax_min
        feats['shape_axis_ratio']    = ax_maj / (ax_min + eps)
    else:
        feats.update({
            'shape_solidity': 0.0, 'shape_extent': 0.0,
            'shape_axis_major_mm': 0.0, 'shape_axis_minor_mm': 0.0,
            'shape_axis_ratio': 0.0,
        })

    # ── E: Spatial Context ───────────────────────────────────────────────────
    cz, cy, cx = candidate['centroid']
    Z, Y, X    = vessel_map.shape

    feats['spatial_z_mm'] = float(cz * VOXEL_SPACING[0])
    feats['spatial_y_mm'] = float(cy * VOXEL_SPACING[1])
    feats['spatial_x_mm'] = float(cx * VOXEL_SPACING[2])

    # BUG FIX 4: depth ratio in mm-space (same value for isotropic, semantically correct)
    feats['spatial_depth_ratio'] = float(
        (cz * VOXEL_SPACING[0]) / (Z * VOXEL_SPACING[0] + eps)
    )

    dist_to_edge = min(cz, Z - cz, cy, Y - cy, cx, X - cx) * VOXEL_SPACING[0]
    feats['spatial_dist_edge_mm'] = float(dist_to_edge)

    # ── F: Detection Metadata ────────────────────────────────────────────────
    # BUG FIX 5: encode method as binary flag (string columns break ML pipelines)
    feats['meta_method_blob_log']  = int(candidate['method'] == 'blob_log')
    feats['meta_candidate_id']     = lid
    feats['meta_peak_vesselness']  = candidate['peak_vesselness']
    feats['meta_mean_vesselness']  = candidate['mean_vesselness']

    return feats


# =============================================================================
# 5.  MAIN PIPELINE
# =============================================================================

def run_pipeline(
    preprocessed_dir:   Path,
    train_csv:          Path,
    localizer_csv:      Path,
    output_path:        Path,
    limit:              int  = None,
    drop_ambiguous:     bool = False,
    ambiguous_as_pos:   bool = False,
    label_radius_mm:    float = LABEL_RADIUS_MM,
):
    log.info("=" * 65)
    log.info("STAGE 1: Discovering cases")
    cases = discover_cases(preprocessed_dir, train_csv, localizer_csv)
    if limit:
        cases = cases[:limit]
        log.info(f"  Limiting to {limit} cases")

    all_rows = []
    n_cases  = len(cases)

    for case_idx, case in enumerate(cases):
        uid           = case['uid']
        nii_path      = case['nii_path']
        has_aneurysm  = case['has_aneurysm']
        loc_flags     = case['location_flags']
        patient_age   = case['patient_age']
        patient_sex   = case['patient_sex']
        loc_xy_px     = case['loc_xy_px']

        log.info(f"\n[{case_idx+1}/{n_cases}] {uid}  aneurysm={has_aneurysm}  "
                 f"n_locs={len(loc_xy_px)}")

        try:
            img        = nib.load(str(nii_path))
            vessel_map = img.get_fdata().astype(np.float32)
            log.info(f"  Shape: {vessel_map.shape} | "
                     f"Range: [{vessel_map.min():.3f}, {vessel_map.max():.3f}]")

            if vessel_map.max() < 1e-6:
                log.warning(f"  [SKIP] Vesselness map is all zeros — {uid}")
                continue

            # STAGE 2: Candidate Generation (BUG FIX 1 applied inside)
            label_map, candidates = generate_candidates(vessel_map)
            log.info(f"  Candidates: {len(candidates)}")
            if not candidates:
                log.warning(f"  [SKIP] No candidates found — {uid}")
                continue

            # NEW — Candidate-level label assignment
            candidates = assign_candidate_labels(
                candidates     = candidates,
                has_aneurysm   = has_aneurysm,
                loc_xy_px      = loc_xy_px,
                nii_shape      = vessel_map.shape,
                label_radius_mm= label_radius_mm,
                ambiguous_as_pos = ambiguous_as_pos,
            )

            pos_cands = sum(1 for c in candidates if c.get('cand_label') == 1)
            amb_cands = sum(1 for c in candidates if c.get('cand_label') == -1)
            log.info(f"  Labeled: {pos_cands} positive, {amb_cands} ambiguous, "
                     f"{len(candidates)-pos_cands-amb_cands} negative")

            # STAGE 3: Feature Extraction
            for cand in candidates:
                cand_label = cand.get('cand_label', -1)

                # Optionally drop ambiguous candidates
                if drop_ambiguous and cand_label == -1:
                    continue

                try:
                    feats = extract_features(cand, vessel_map, label_map)
                except Exception as e:
                    log.warning(f"  Feature extraction failed for candidate "
                                f"{cand['label_id']}: {e}")
                    continue

                # ── Candidate-level and case-level labels ─────────────────
                feats['cand_label']    = cand_label
                feats['case_label']    = has_aneurysm
                feats['case_id']       = uid

                # ── XY distance to nearest localizer point ─────────────────
                feats['xy_dist_to_nearest_loc_mm'] = cand.get(
                    'xy_dist_to_nearest_loc_mm', float('nan')
                )

                # ── Patient demographics ───────────────────────────────────
                feats['patient_age']      = patient_age
                feats['patient_sex_male'] = int(patient_sex.strip().lower() == 'male')

                # ── Artery location flags (case-level) ────────────────────
                for col, val in loc_flags.items():
                    feats[f'loc_{col.replace(" ", "_").lower()}'] = val

                all_rows.append(feats)

        except Exception as e:
            log.warning(f"  [FAILED] {uid}: {e}")
            continue

    if not all_rows:
        log.error("No features extracted — check paths and preprocessing outputs.")
        return None

    # STAGE 4: Export
    log.info("\n" + "=" * 65)
    df = pd.DataFrame(all_rows)

    # Reorder: identifiers → labels → distance → demographics → features
    id_cols = [
        'case_id', 'cand_label', 'case_label',
        'xy_dist_to_nearest_loc_mm',
        'meta_candidate_id', 'meta_method_blob_log',
        'meta_peak_vesselness', 'meta_mean_vesselness',
        'patient_age', 'patient_sex_male',
    ]
    loc_label_cols = [c for c in df.columns if c.startswith('loc_')]
    id_cols        = [c for c in id_cols if c in df.columns]
    feat_cols      = [c for c in df.columns
                      if c not in id_cols and c not in loc_label_cols]
    df = df[id_cols + loc_label_cols + feat_cols]

    df.to_csv(output_path, index=False)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    log.info(f"Saved: {len(df)} candidates × {len(num_cols)} numeric features")
    log.info(f"Output: {output_path}")
    log.info(f"Cases processed: {df['case_id'].nunique()} / {n_cases}")

    log.info("\nCandidate-level label distribution:")
    log.info(df['cand_label'].value_counts().to_string())

    log.info("\nCase-level distribution (cases with ≥1 candidate):")
    log.info(df.groupby('case_label')['case_id'].nunique().to_string())

    log.info("\nFeature statistics preview (first 8 numeric features):")
    log.info(df[num_cols[:8]].describe().round(4).to_string())

    return df


# =============================================================================
# 6.  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature extraction with candidate-level label assignment "
                    "(Frangi vesselness maps + localizer coordinates)"
    )
    parser.add_argument("--preprocessed_dir",  required=True,
        help="Root dir of preprocessed outputs (one sub-folder per case).")
    parser.add_argument("--train_csv",         required=True,
        help="Path to balanced_mra_train.csv (SeriesInstanceUID + labels).")
    parser.add_argument("--localizer_csv",     required=True,
        help="Path to train_localizers.csv (x,y aneurysm coordinates).")
    parser.add_argument("--output",            default="features_v2.csv",
        help="Output CSV path. Default: features_v2.csv")
    parser.add_argument("--limit",             type=int,  default=None,
        help="Process only the first N cases (debugging).")
    parser.add_argument("--label_radius_mm",   type=float, default=LABEL_RADIUS_MM,
        help=f"XY distance threshold for positive candidate labeling "
             f"(default: {LABEL_RADIUS_MM} mm).")
    parser.add_argument("--drop_ambiguous",    action="store_true",
        help="Drop candidates from positive cases that have no localizer entry "
             "(cand_label == -1). Recommended for clean training sets.")
    parser.add_argument("--ambiguous_as_positive", action="store_true",
        help="Treat ambiguous candidates (positive case, no loc coords) as "
             "cand_label=1 instead of -1. Conservative recall-first strategy.")
    args = parser.parse_args()

    run_pipeline(
        preprocessed_dir  = Path(args.preprocessed_dir),
        train_csv         = Path(args.train_csv),
        localizer_csv     = Path(args.localizer_csv),
        output_path       = Path(args.output),
        limit             = args.limit,
        drop_ambiguous    = args.drop_ambiguous,
        ambiguous_as_pos  = args.ambiguous_as_positive,
        label_radius_mm   = args.label_radius_mm,
    )