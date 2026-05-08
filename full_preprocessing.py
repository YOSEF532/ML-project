import subprocess, sys, os, shutil, warnings, logging, traceback
warnings.filterwarnings("ignore")

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("nibabel", "SimpleITK", "scikit-image", "hd-bet", "dicom2nifti")

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import dicom2nifti
import dicom2nifti.settings as d2n_settings

from pathlib import Path
from skimage.filters import frangi
import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════


# Use 'r' before the string to handle backslashes (raw string)

CSV_PATH      = Path(r'C:\Cairo university\Second Year\Machine Learning\mra_subset_dataset\balanced_mra_train.csv')

SERIES_FOLDER = Path(r'C:\Cairo university\Second Year\Machine Learning\mra_subset_dataset\series')

NIFTI_RAW_DIR = Path(r'C:\Cairo university\Second Year\Machine Learning\mra_subset_dataset\nifti_raw')



# Fixed the leading slash and added 'r'

OUTPUT_DIR    = Path(r'C:\Cairo university\Second Year\Machine Learning\preprocessed')

TEMP_DIR      = Path(r'C:\Cairo university\Second Year\Machine Learning\_tmp')        # ← shared temp root; wiped per series

for d in [NIFTI_RAW_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_SPACING       = (0.5, 0.5, 0.5)
USE_HD_BET           = False
N4_ITERS             = [20, 20, 10]
VESSEL_SIGMAS        = [1.0, 2.0, 3.0]
VESSEL_ALPHA         = 0.5
VESSEL_BETA          = 0.5
VESSEL_GAMMA         = 5.0
VESSEL_BRIGHT_OBJECT = True

d2n_settings.disable_validate_slice_increment()
d2n_settings.disable_validate_orientation()

# ══════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════
log_file = OUTPUT_DIR / 'pipeline.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file), mode='w')
    ]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
#  STAGE 1 — LOAD CSV & BUILD LABEL MAP
# ══════════════════════════════════════════════════════════════════
df_csv = pd.read_csv(CSV_PATH)

label_cols = [c for c in df_csv.columns
              if c not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality']]

available_series = set(os.listdir(SERIES_FOLDER))
df_csv['on_disk'] = df_csv['SeriesInstanceUID'].astype(str).isin(available_series)
df_avail = df_csv[df_csv['on_disk']].copy()
df_avail['has_aneurysm'] = (df_avail[label_cols].sum(axis=1) > 0).astype(int)

label_map = dict(zip(df_avail['SeriesInstanceUID'].astype(str),
                     df_avail['has_aneurysm']))

log.info(f"Series on disk   : {len(df_avail)}")
log.info(f"With aneurysm    : {df_avail['has_aneurysm'].sum()}")
log.info(f"Without aneurysm : {(df_avail['has_aneurysm'] == 0).sum()}")

# ══════════════════════════════════════════════════════════════════
#  STAGE 2 — DICOM → NIfTI
# ══════════════════════════════════════════════════════════════════

def dicom_to_nifti(uid: str, dicom_dir: Path, nifti_dir: Path) -> Path | None:
    out_path = nifti_dir / f"{uid}.nii.gz"
    if out_path.exists():
        log.info(f"[DICOM→NII][SKIP] {uid} — already converted")
        return out_path

    dcm_files = list(dicom_dir.glob('*.dcm')) or list(dicom_dir.iterdir())
    if not dcm_files:
        log.warning(f"[DICOM→NII][EMPTY] {uid} — no slices found")
        return None

    tmp_dir = nifti_dir / f"_tmp_{uid}"
    tmp_dir.mkdir(exist_ok=True)

    try:
        dicom2nifti.convert_directory(
            dicom_directory=str(dicom_dir),
            output_folder=str(tmp_dir),
            compression=True,
            reorient=True,
        )
        candidates = sorted(tmp_dir.glob('*.nii.gz'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            log.error(f"[DICOM→NII][FAIL] {uid} — conversion produced no output")
            return None

        candidates[-1].rename(out_path)
        log.info(f"[DICOM→NII][OK] {uid}")
        return out_path

    except Exception as e:
        log.error(f"[DICOM→NII][FAIL] {uid} — {e}")
        return None

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)     # ← always clean up


# ══════════════════════════════════════════════════════════════════
#  STAGE 3 — PREPROCESSING HELPERS  (unchanged)
# ══════════════════════════════════════════════════════════════════

def resample_to_spacing(sitk_img, new_spacing=TARGET_SPACING,
                        interpolator=sitk.sitkLinear):
    orig_spacing = np.array(sitk_img.GetSpacing(), dtype=np.float32)
    orig_size    = np.array(sitk_img.GetSize(),    dtype=np.int32)
    new_size     = np.round(orig_size * orig_spacing / np.array(new_spacing)).astype(int).tolist()
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(new_spacing)
    r.SetSize(new_size)
    r.SetOutputDirection(sitk_img.GetDirection())
    r.SetOutputOrigin(sitk_img.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(interpolator)
    return r.Execute(sitk_img)


def n4_bias_correction(sitk_img, n_iters=N4_ITERS):
    img_f     = sitk.Cast(sitk_img, sitk.sitkFloat32)
    mask_img  = sitk.OtsuThreshold(img_f, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_iters)
    return corrector.Execute(img_f, mask_img)


def skull_strip_hdbet(nii_path: Path, out_dir: Path) -> Path:
    out_path = out_dir / (nii_path.stem.replace('.nii', '') + '_brain.nii.gz')
    result = subprocess.run(
        ['hd-bet', '-i', str(nii_path), '-o', str(out_path),
         '-device', 'cpu', '-mode', 'fast', '-tta', '0'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"hd-bet failed (code {result.returncode}):\n{result.stderr.strip()}")
    return out_path


def skull_strip_sitk_fallback(sitk_img):
    img_f   = sitk.Cast(sitk_img, sitk.sitkFloat32)
    otsu    = sitk.OtsuThreshold(img_f, 0, 1, 200)
    filled  = sitk.BinaryFillhole(otsu)
    eroded  = sitk.BinaryErode(filled,  [3, 3, 3])
    dilated = sitk.BinaryDilate(eroded, [2, 2, 2])
    cc      = sitk.ConnectedComponent(dilated)
    relabel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    brain_mask = sitk.BinaryThreshold(relabel, 1, 1, 1, 0)
    mask_arr   = sitk.GetArrayFromImage(brain_mask).astype(np.uint8)
    img_arr    = sitk.GetArrayFromImage(img_f)
    masked_arr = (img_arr * mask_arr).astype(np.float32)
    masked_sitk = sitk.GetImageFromArray(masked_arr)
    masked_sitk.CopyInformation(img_f)
    return masked_sitk, mask_arr


def vesselness_filter(sitk_img):
    img_f = sitk.Cast(sitk_img, sitk.sitkFloat32)
    arr   = sitk.GetArrayFromImage(img_f)
    vessel_arr = frangi(
        arr,
        sigmas=VESSEL_SIGMAS,
        alpha=VESSEL_ALPHA,
        beta=VESSEL_BETA,
        gamma=VESSEL_GAMMA,
        black_ridges=not VESSEL_BRIGHT_OBJECT,
    ).astype(np.float32)
    v_max = vessel_arr.max()
    if v_max > 1e-8:
        vessel_arr /= v_max
    vessel_sitk = sitk.GetImageFromArray(vessel_arr)
    vessel_sitk.CopyInformation(img_f)
    return vessel_sitk


def zscore_normalise(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    brain = arr[mask > 0]
    if brain.size == 0:
        return arr.astype(np.float32)
    mu, sd = brain.mean(), brain.std()
    if sd < 1e-8:
        return arr.astype(np.float32)
    normed = ((arr - mu) / sd).astype(np.float32)
    normed[mask == 0] = 0.0
    return normed


# ══════════════════════════════════════════════════════════════════
#  STAGE 3 — PREPROCESS ONE NIfTI  (temp-folder edition)
# ══════════════════════════════════════════════════════════════════

def preprocess_nifti(nifti_path: Path, uid: str, output_dir: Path) -> dict:

    # ── Permanent output folder (only the final file ends up here) ──
    subj_dir = output_dir / uid
    subj_dir.mkdir(exist_ok=True)

    final_nii = subj_dir / f'{uid}_final.nii.gz'
    if final_nii.exists():
        log.info(f"[PREPROCESS][SKIP] {uid} — already preprocessed")
        return {
            'case_id'      : uid,
            'has_aneurysm' : label_map.get(uid, -1),
            'status'       : 'ok',
            'final_path'   : str(final_nii),
            'errors'       : [], }

    # ── Per-series temp folder (all intermediates go here) ─────────
    tmp = TEMP_DIR / uid
    tmp.mkdir(exist_ok=True)

    
    meta = {
        'case_id'            : uid,
        'has_aneurysm'       : label_map.get(uid, -1),
        'status'             : 'ok',
        'skull_strip_method' : None,
        'errors'             : [],
    }


    try:
        # ── 1. Load ──────────────────────────────────────────────────
        log.info(f"  [{uid}] 1/6 Loading NIfTI...")
        sitk_img = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
        meta['original_spacing'] = tuple(round(s, 4) for s in sitk_img.GetSpacing())
        meta['original_size']    = sitk_img.GetSize()

        # ── 2. Resample → temp ────────────────────────────────────────
        log.info(f"  [{uid}] 2/6 Resampling → {TARGET_SPACING} mm...")
        resampled = resample_to_spacing(sitk_img)
        sitk.WriteImage(resampled, str(tmp / f'{uid}_resampled.nii.gz'))

        # ── 3. N4 bias correction → temp ──────────────────────────────
        log.info(f"  [{uid}] 3/6 N4 bias correction...")
        n4_img  = n4_bias_correction(resampled)
        n4_nii  = tmp / f'{uid}_n4.nii.gz'
        sitk.WriteImage(n4_img, str(n4_nii))

        # ── 4. Skull stripping → temp ─────────────────────────────────
        if USE_HD_BET:
            log.info(f"  [{uid}] 4/6 Skull stripping (HD-BET)...")
            try:
                stripped_nii  = skull_strip_hdbet(n4_nii, tmp)   # ← tmp, not subj_dir
                stripped_sitk = sitk.ReadImage(str(stripped_nii), sitk.sitkFloat32)
                mask_arr      = (sitk.GetArrayFromImage(stripped_sitk) > 0).astype(np.uint8)
                meta['skull_strip_method'] = 'hd-bet'
            except RuntimeError as e:
                meta['errors'].append(str(e))
                log.warning(f"  [{uid}] HD-BET failed → Otsu fallback")
                stripped_sitk, mask_arr    = skull_strip_sitk_fallback(n4_img)
                meta['skull_strip_method'] = 'otsu-fallback'
        else:
            log.info(f"  [{uid}] 4/6 Skull stripping (Otsu fallback)...")
            stripped_sitk, mask_arr    = skull_strip_sitk_fallback(n4_img)
            meta['skull_strip_method'] = 'otsu-fallback'

        # Write stripped to temp (needed for affine extraction below)
        stripped_tmp = tmp / f'{uid}_stripped.nii.gz'
        sitk.WriteImage(stripped_sitk, str(stripped_tmp))

        # ── 5. Z-score normalise (in memory) ─────────────────────────
        log.info(f"  [{uid}] 5/6 Z-score normalising...")
        brain_arr   = sitk.GetArrayFromImage(stripped_sitk)
        normed_arr  = zscore_normalise(brain_arr, mask_arr)
        normed_sitk = sitk.GetImageFromArray(normed_arr)
        normed_sitk.CopyInformation(stripped_sitk)

        # ── 6. Vesselness (Frangi) (in memory) ───────────────────────
        log.info(f"  [{uid}] 6/6 Vesselness filter (Frangi)...")
        vessel_sitk = vesselness_filter(normed_sitk)
        vessel_arr  = sitk.GetArrayFromImage(vessel_sitk)

        # ── Save ONLY the final file to the permanent output folder ───
        img_nib   = nib.load(str(stripped_tmp))
        final_nii = subj_dir / f'{uid}_final.nii.gz'
        nib.save(
            nib.Nifti1Image(vessel_arr, img_nib.affine, img_nib.header),
            str(final_nii)
        )

        meta['final_path']   = str(final_nii)
        meta['final_shape']  = vessel_arr.shape
        meta['brain_voxels'] = int(mask_arr.sum())

        log.info(f"  [{uid}] ✅ Done — shape={vessel_arr.shape}, "
                 f"brain_voxels={meta['brain_voxels']:,}")

    except Exception as exc:
        meta['status'] = 'FAILED'
        meta['errors'].append(traceback.format_exc())
        log.error(f"  [{uid}] ❌ FAILED: {exc}")

    finally:
        # ── Always wipe the temp folder for this series ───────────────
        shutil.rmtree(tmp, ignore_errors=True)
        log.info(f"  [{uid}] 🗑  Temp folder removed")

    return meta


# ══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════
all_results = []
uids  = df_avail['SeriesInstanceUID'].astype(str).tolist()
total = len(uids)

for i, uid in enumerate(uids, 1):
    log.info(f"\n{'═'*60}")
    log.info(f"  Series {i}/{total} | UID: {uid} | label: {label_map[uid]}")
    log.info(f"{'═'*60}")

    dicom_dir = SERIES_FOLDER / uid
    nifti_raw = dicom_to_nifti(uid, dicom_dir, NIFTI_RAW_DIR)

    if nifti_raw is None:
        all_results.append({
            'case_id'      : uid,
            'has_aneurysm' : label_map.get(uid, -1),
            'status'       : 'FAILED',
            'errors'       : ['DICOM→NIfTI conversion failed'],
        })
        continue

    meta = preprocess_nifti(nifti_raw, uid, OUTPUT_DIR)
    all_results.append(meta)

# ══════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
df_results = pd.DataFrame(all_results)
ok_count   = (df_results['status'] == 'ok').sum()
fail_count = (df_results['status'] == 'FAILED').sum()

log.info(f"\n{'═'*60}")
log.info(f"  PIPELINE COMPLETE")
log.info(f"  ✅ Succeeded : {ok_count}")
log.info(f"  ❌ Failed    : {fail_count}")
log.info(f"{'═'*60}")

summary_path = OUTPUT_DIR / 'pipeline_summary.csv'
df_results.to_csv(summary_path, index=False)
log.info(f"\n📋 Summary saved → {summary_path}")
print(df_results[['case_id', 'has_aneurysm', 'status', 'final_shape', 'brain_voxels']].to_string())