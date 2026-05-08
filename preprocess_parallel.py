"""
preprocess_parallel.py  --  Main pipeline orchestrator.

Architecture:
  Stage 2  DICOM -> NIfTI   : ThreadPoolExecutor (I/O-bound, 8 threads)
  Stage 3  Preprocessing    : ThreadPoolExecutor launching subprocess per case
                              (each subprocess is a fully isolated Python process
                               running preprocess_worker.py -- no pickling issues,
                               no shared memory, automatic cleanup on crash)

IMPORTANT: preprocess_worker.py must be in the SAME folder as this script.
"""

import subprocess, sys, os, shutil, warnings, logging, traceback, json
warnings.filterwarnings("ignore")

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip_install("nibabel", "SimpleITK", "scikit-image", "hd-bet", "dicom2nifti")

import pandas as pd
import dicom2nifti
import dicom2nifti.settings as d2n_settings

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================================================================
#  CONFIG
# ================================================================

CSV_PATH      = Path(r'C:\Cairo university\Second Year\Machine Learning\mra_subset_dataset\balanced_mra_train.csv')
SERIES_FOLDER = Path(r'C:\Cairo university\Second Year\Machine Learning\mra_subset_dataset\series')
NIFTI_RAW_DIR = Path(r'C:\Cairo university\Second Year\Machine Learning\mra_subset_dataset\nifti_raw')
OUTPUT_DIR    = Path(r'C:\Cairo university\Second Year\Machine Learning\preprocessed')
TEMP_DIR      = Path(r'C:\Cairo university\Second Year\Machine Learning\_tmp')

# Worker script -- must live next to this file
WORKER_SCRIPT = Path(__file__).parent / 'preprocess_worker.py'

for d in [NIFTI_RAW_DIR, OUTPUT_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- Parallelism tuning for i9 Ultra 185H ----------------------
# Each subprocess is fully isolated: imports, memory, everything.
# No pickling. No shared state. Crashes are contained per-case.
#
# RAM guide  (each subprocess peaks ~3-4 GB):
#   32 GB RAM -> use 6
#   64 GB RAM -> use 12
DICOM_THREADS      = 8    # DICOM->NIfTI threads (I/O bound)
PREPROCESS_WORKERS = 6    # parallel preprocessing subprocesses

# Timeout per case in seconds (MRA N4+Frangi can take 5-15 min)
WORKER_TIMEOUT_SEC = 900  # 15 minutes

d2n_settings.disable_validate_slice_increment()
d2n_settings.disable_validate_orientation()

# ================================================================
#  LOGGING
# ================================================================
log_file = OUTPUT_DIR / 'pipeline.log'
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
_fh = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[_sh, _fh])
log = logging.getLogger(__name__)
SEP = '=' * 60

# ================================================================
#  STAGE 1 -- CSV + LABEL MAP
# ================================================================
df_csv = pd.read_csv(CSV_PATH)
label_cols = [c for c in df_csv.columns
              if c not in ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality']]

available_series = set(os.listdir(SERIES_FOLDER))
df_csv['on_disk'] = df_csv['SeriesInstanceUID'].astype(str).isin(available_series)
df_avail = df_csv[df_csv['on_disk']].copy()
df_avail['has_aneurysm'] = (df_avail[label_cols].sum(axis=1) > 0).astype(int)
label_map = dict(zip(df_avail['SeriesInstanceUID'].astype(str), df_avail['has_aneurysm']))

log.info(f"Series on disk    : {len(df_avail)}")
log.info(f"With aneurysm     : {df_avail['has_aneurysm'].sum()}")
log.info(f"Without aneurysm  : {(df_avail['has_aneurysm'] == 0).sum()}")
log.info(f"DICOM threads     : {DICOM_THREADS}")
log.info(f"Preprocess workers: {PREPROCESS_WORKERS}")
log.info(f"Worker script     : {WORKER_SCRIPT}")

if not WORKER_SCRIPT.exists():
    log.error(f"Worker script not found: {WORKER_SCRIPT}")
    log.error("Put preprocess_worker.py in the same folder as this script.")
    sys.exit(1)

# ================================================================
#  STAGE 2 -- DICOM -> NIfTI  (threaded, I/O bound)
# ================================================================

def dicom_to_nifti(uid, dicom_dir, nifti_dir):
    out_path = nifti_dir / f"{uid}.nii.gz"
    if out_path.exists():
        log.info(f"[DICOM->NII][SKIP] {uid}")
        return uid, out_path

    dcm_files = list(dicom_dir.glob('*.dcm')) or list(dicom_dir.iterdir())
    if not dcm_files:
        log.warning(f"[DICOM->NII][EMPTY] {uid}")
        return uid, None

    tmp_dir = nifti_dir / f"_tmp_{uid}"
    tmp_dir.mkdir(exist_ok=True)
    try:
        dicom2nifti.convert_directory(
            dicom_directory=str(dicom_dir),
            output_folder=str(tmp_dir),
            compression=True, reorient=True,
        )
        candidates = sorted(tmp_dir.glob('*.nii.gz'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            log.error(f"[DICOM->NII][FAIL] {uid} - no output")
            return uid, None
        candidates[-1].rename(out_path)
        log.info(f"[DICOM->NII][OK] {uid}")
        return uid, out_path
    except Exception as e:
        log.error(f"[DICOM->NII][FAIL] {uid} - {e}")
        return uid, None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_dicom_stage(uids):
    results = {}
    log.info(f"\n{SEP}\n  STAGE 2: DICOM -> NIfTI  ({DICOM_THREADS} threads)\n{SEP}")
    with ThreadPoolExecutor(max_workers=DICOM_THREADS) as pool:
        futures = {pool.submit(dicom_to_nifti, uid, SERIES_FOLDER / uid, NIFTI_RAW_DIR): uid
                   for uid in uids}
        done = 0
        for fut in as_completed(futures):
            uid, nii_path = fut.result()
            results[uid] = nii_path
            done += 1
            if done % 10 == 0 or done == len(uids):
                log.info(f"  DICOM->NII progress: {done}/{len(uids)}")
    ok = sum(1 for v in results.values() if v is not None)
    log.info(f"  DICOM->NII complete: {ok}/{len(uids)} succeeded")
    return results


# ================================================================
#  STAGE 3 -- PREPROCESS via SUBPROCESSES
#  No pickling, no ProcessPoolExecutor, no OOM cascade.
#  Each case = one isolated python subprocess.
# ================================================================

def run_one_subprocess(uid, nifti_path, label):
    result_json = TEMP_DIR / f"_result_{uid}.json"

    cmd = [
        sys.executable, str(WORKER_SCRIPT),
        uid,
        str(nifti_path),
        str(OUTPUT_DIR),
        str(TEMP_DIR),
        str(label),
        str(result_json),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=WORKER_TIMEOUT_SEC,
        )

        if proc.stdout.strip():
            for line in proc.stdout.strip().splitlines():
                log.info(f"  [worker] {line}")
        if proc.stderr.strip():
            for line in proc.stderr.strip().splitlines():
                log.warning(f"  [worker-err] {line}")

        if result_json.exists():
            meta = json.loads(result_json.read_text(encoding='utf-8'))
            result_json.unlink(missing_ok=True)
            return meta
        else:
            return {
                'case_id': uid, 'has_aneurysm': label, 'status': 'FAILED',
                'errors': [
                    f"Worker exited (rc={proc.returncode}) with no result file.\n"
                    f"STDERR (last 2000 chars):\n{proc.stderr[-2000:]}"
                ],
            }

    except subprocess.TimeoutExpired:
        log.error(f"  [worker] {uid} TIMED OUT after {WORKER_TIMEOUT_SEC}s")
        return {'case_id': uid, 'has_aneurysm': label, 'status': 'FAILED',
                'errors': [f"Timeout after {WORKER_TIMEOUT_SEC}s"]}
    except Exception as exc:
        return {'case_id': uid, 'has_aneurysm': label, 'status': 'FAILED',
                'errors': [traceback.format_exc()]}
    finally:
        result_json.unlink(missing_ok=True)


def run_preprocess_stage(nifti_map):
    log.info(f"\n{SEP}\n  STAGE 3: PREPROCESSING  ({PREPROCESS_WORKERS} subprocesses)\n{SEP}")

    jobs, skipped = [], []
    for uid, nii_path in nifti_map.items():
        if nii_path is None:
            skipped.append({'case_id': uid, 'has_aneurysm': label_map.get(uid, -1),
                            'status': 'FAILED', 'errors': ['DICOM->NIfTI failed']})
        else:
            jobs.append((uid, nii_path, label_map.get(uid, -1)))

    results = list(skipped)
    total, done = len(jobs), 0

    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
        futures = {pool.submit(run_one_subprocess, uid, nii, lbl): uid
                   for uid, nii, lbl in jobs}
        for fut in as_completed(futures):
            uid = futures[fut]
            try:
                meta = fut.result()
            except Exception:
                meta = {'case_id': uid, 'has_aneurysm': label_map.get(uid, -1),
                        'status': 'FAILED', 'errors': [traceback.format_exc()]}
            results.append(meta)
            done += 1
            status_str = meta.get('status', '?')
            shape_str  = str(meta.get('final_shape', ''))
            log.info(f"  [{done}/{total}] [{status_str}] {uid[:40]}  {shape_str}")

    return results


# ================================================================
#  MAIN
# ================================================================
if __name__ == '__main__':
    uids = df_avail['SeriesInstanceUID'].astype(str).tolist()
    log.info(f"Total series to process: {len(uids)}")

    nifti_map   = run_dicom_stage(uids)
    all_results = run_preprocess_stage(nifti_map)

    df_results = pd.DataFrame(all_results)
    ok_count   = (df_results['status'] == 'ok').sum()
    fail_count = (df_results['status'] == 'FAILED').sum()

    log.info(f"\n{SEP}")
    log.info(f"  PIPELINE COMPLETE")
    log.info(f"  [OK]     Succeeded : {ok_count}")
    log.info(f"  [FAILED] Failed    : {fail_count}")
    log.info(SEP)

    summary_path = OUTPUT_DIR / 'pipeline_summary.csv'
    df_results.to_csv(summary_path, index=False)
    log.info(f"Summary saved -> {summary_path}")

    display_cols = [c for c in ['case_id', 'has_aneurysm', 'status', 'final_shape', 'brain_voxels']
                    if c in df_results.columns]
    print(df_results[display_cols].to_string())
