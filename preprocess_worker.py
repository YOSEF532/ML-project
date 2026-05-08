"""
preprocess_worker.py  —  processes ONE NIfTI case.
Called by preprocess_parallel.py as a subprocess:
    python preprocess_worker.py <uid> <nifti_path> <output_dir> <temp_dir> <label> <result_json_path>

Writes a JSON result file on exit so the parent can read the outcome.
All stdout/stderr is captured by the parent.
"""
import sys, os, shutil, warnings, traceback, json
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from skimage.filters import frangi

# ── config passed in by parent via env vars ──────────────────────
TARGET_SPACING       = (0.5, 0.5, 0.5)
N4_ITERS             = [20, 20, 10]
VESSEL_SIGMAS        = [1.0, 2.0, 3.0]
VESSEL_ALPHA         = 0.5
VESSEL_BETA          = 0.5
VESSEL_GAMMA         = 5.0
VESSEL_BRIGHT_OBJECT = True


def resample_to_spacing(sitk_img, new_spacing=TARGET_SPACING, interpolator=sitk.sitkLinear):
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
        arr, sigmas=VESSEL_SIGMAS, alpha=VESSEL_ALPHA,
        beta=VESSEL_BETA, gamma=VESSEL_GAMMA,
        black_ridges=not VESSEL_BRIGHT_OBJECT,
    ).astype(np.float32)
    v_max = vessel_arr.max()
    if v_max > 1e-8:
        vessel_arr /= v_max
    vessel_sitk = sitk.GetImageFromArray(vessel_arr)
    vessel_sitk.CopyInformation(img_f)
    return vessel_sitk


def zscore_normalise(arr, mask):
    brain = arr[mask > 0]
    if brain.size == 0:
        return arr.astype(np.float32)
    mu, sd = brain.mean(), brain.std()
    if sd < 1e-8:
        return arr.astype(np.float32)
    normed = ((arr - mu) / sd).astype(np.float32)
    normed[mask == 0] = 0.0
    return normed


def run(uid, nifti_path_str, output_dir_str, temp_dir_str, label):
    nifti_path = Path(nifti_path_str)
    output_dir = Path(output_dir_str)
    temp_dir   = Path(temp_dir_str)

    subj_dir  = output_dir / uid
    subj_dir.mkdir(parents=True, exist_ok=True)
    final_nii = subj_dir / f'{uid}_final.nii.gz'

    if final_nii.exists():
        print(f"[SKIP] {uid} already done", flush=True)
        return {'case_id': uid, 'has_aneurysm': label, 'status': 'ok',
                'final_path': str(final_nii), 'errors': []}

    tmp = temp_dir / uid
    tmp.mkdir(parents=True, exist_ok=True)

    meta = {'case_id': uid, 'has_aneurysm': label, 'status': 'ok', 'errors': []}

    try:
        print(f"[{uid}] 1/6 Loading...", flush=True)
        sitk_img = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
        meta['original_spacing'] = list(round(s, 4) for s in sitk_img.GetSpacing())
        meta['original_size']    = list(sitk_img.GetSize())

        print(f"[{uid}] 2/6 Resampling...", flush=True)
        resampled = resample_to_spacing(sitk_img)
        sitk.WriteImage(resampled, str(tmp / f'{uid}_resampled.nii.gz'))

        print(f"[{uid}] 3/6 N4 bias correction...", flush=True)
        n4_img = n4_bias_correction(resampled)
        n4_nii = tmp / f'{uid}_n4.nii.gz'
        sitk.WriteImage(n4_img, str(n4_nii))

        print(f"[{uid}] 4/6 Skull stripping...", flush=True)
        stripped_sitk, mask_arr = skull_strip_sitk_fallback(n4_img)
        stripped_tmp = tmp / f'{uid}_stripped.nii.gz'
        sitk.WriteImage(stripped_sitk, str(stripped_tmp))

        print(f"[{uid}] 5/6 Z-score normalising...", flush=True)
        brain_arr   = sitk.GetArrayFromImage(stripped_sitk)
        normed_arr  = zscore_normalise(brain_arr, mask_arr)
        normed_sitk = sitk.GetImageFromArray(normed_arr)
        normed_sitk.CopyInformation(stripped_sitk)

        print(f"[{uid}] 6/6 Vesselness (Frangi)...", flush=True)
        vessel_sitk = vesselness_filter(normed_sitk)
        vessel_arr  = sitk.GetArrayFromImage(vessel_sitk)

        img_nib = nib.load(str(stripped_tmp))
        nib.save(nib.Nifti1Image(vessel_arr, img_nib.affine, img_nib.header), str(final_nii))

        meta['final_path']   = str(final_nii)
        meta['final_shape']  = list(vessel_arr.shape)
        meta['brain_voxels'] = int(mask_arr.sum())
        print(f"[{uid}] DONE shape={vessel_arr.shape} brain_voxels={meta['brain_voxels']}", flush=True)

    except Exception as exc:
        meta['status'] = 'FAILED'
        meta['errors'].append(traceback.format_exc())
        print(f"[{uid}] FAILED: {exc}", flush=True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return meta


if __name__ == '__main__':
    # argv: uid nifti_path output_dir temp_dir label result_json
    uid, nifti_path, output_dir, temp_dir, label_str, result_json = sys.argv[1:]
    result = run(uid, nifti_path, output_dir, temp_dir, int(label_str))
    Path(result_json).write_text(json.dumps(result), encoding='utf-8')
    sys.exit(0 if result['status'] == 'ok' else 1)
