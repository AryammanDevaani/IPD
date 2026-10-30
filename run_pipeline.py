r"""Run the full ScriptSense training pipeline locally.

This script replicates what the notebook does in sections 1-14,
but runs directly as a Python script without Jupyter.
"""
import os
import sys
import pathlib
import subprocess
import shutil
import json

# ── Setup ───────────────────────────────────────────────────────────────────
WORK_DIR = pathlib.Path.home() / "scriptsense_notebook"
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PY = sys.executable
DATA_DIR = pathlib.Path(os.environ.get("SCRIPTSENSE_DATA_DIR", pathlib.Path.home() / "scriptsense_data"))
ART_DIR = pathlib.Path(os.environ.get("SCRIPTSENSE_ARTIFACT_DIR", DATA_DIR / "artifacts"))
DATASET = DATA_DIR / "processed" / "dataset.parquet"
ART_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

print("=" * 70)
print("  ScriptSense Full Training Pipeline")
print("=" * 70)
print(f"  Work dir  : {WORK_DIR}")
print(f"  Data dir  : {DATA_DIR}")
print(f"  Model dir : {ART_DIR}")
print(f"  Python    : {PY}")
print()


def sh(*args, cwd=None):
    print(f">>> {PY} {' '.join(str(a) for a in args)}", flush=True)
    p = subprocess.Popen(
        [PY, *args], cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"Command failed (exit {p.returncode}): {' '.join(map(str, args))}")


def run_finished(name):
    meta = ART_DIR / "runs" / name / "run_meta.json"
    return meta.exists() and bool(json.loads(meta.read_text()).get("finished_at"))


# ── Step 1: Create package structure ────────────────────────────────────────
print("\n[1/10] Creating package structure...")
for d in ['ml/scriptsense_ml/data', 'ml/scriptsense_ml/models', 'ml/scriptsense_ml/training',
          'ml/scriptsense_ml/inference', 'ml/scriptsense_ml/evaluation', 'ml/scriptsense_ml/similarity',
          'ml/scripts', 'ml/configs', 'ml/tests', 'data/manifests']:
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
for f in ['scriptsense_ml/__init__.py', 'scriptsense_ml/data/__init__.py',
          'scriptsense_ml/evaluation/__init__.py', 'scriptsense_ml/inference/__init__.py',
          'scriptsense_ml/models/__init__.py', 'scriptsense_ml/similarity/__init__.py',
          'scriptsense_ml/training/__init__.py']:
    pathlib.Path('ml', f).touch()
print("  Done.")


# ── Step 2: Extract code from notebook ──────────────────────────────────────
print("\n[2/10] Extracting code from notebook...")
nb_path = SCRIPT_DIR / "ScriptSense_ML_Pipeline.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
written = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = cell.get("source", [])
        if src and src[0].startswith("%%writefile"):
            filepath = src[0].replace("%%writefile", "").strip()
            target = pathlib.Path(filepath)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(src[1:]), encoding="utf-8")
            written += 1
print(f"  Extracted {written} files.")


# ── Step 3: Verify imports ──────────────────────────────────────────────────
print("\n[3/10] Verifying environment...")
import torch
print(f"  torch {torch.__version__} | CUDA: {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f" | {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    print()
    vram_gb = 0


# ── Step 4: Unit tests ─────────────────────────────────────────────────────
print("\n[4/10] Running unit tests...")
sh("-m", "pytest", "-q", "-p", "no:warnings", cwd="ml")
print("  All tests passed.")


# ── Step 5: Check for existing data ────────────────────────────────────────
DATA_READY = DATASET.exists()
print(f"\n[5/10] Dataset exists: {DATA_READY}")


# ── Step 6: Download datasets ──────────────────────────────────────────────
if not DATA_READY:
    print("\n[6/10] Downloading datasets (TXD-22, SemEval, RAID)...")
    print("  This downloads ~12 GB and may take 30-60 minutes.")
    sh("ml/scripts/download_datasets.py", "txd22", "semeval", "raid_train")
else:
    print("\n[6/10] Skipped (dataset already exists).")


# ── Step 7: Profile datasets ───────────────────────────────────────────────
if not DATA_READY:
    print("\n[7/10] Profiling raw datasets...")
    sh("ml/scripts/inspect_datasets.py", "txd22", "semeval")
else:
    print("\n[7/10] Skipped.")


# ── Step 8: Prepare canonical rows ──────────────────────────────────────────
if not DATA_READY:
    print("\n[8/10] Preparing canonical rows (expect 20-40 min)...")
    sh("ml/scripts/prepare_semeval.py")
    sh("ml/scripts/prepare_txd22.py")
    sh("ml/scripts/prepare_raid.py")
else:
    print("\n[8/10] Skipped.")


# ── Step 9: Dedup, split, leakage audit ─────────────────────────────────────
if not DATA_READY:
    print("\n[9/10] Deduplication, splitting, leakage audit (expect 20-30 min)...")
    sh("ml/scripts/deduplicate_and_split.py")
    DATA_READY = True
else:
    print("\n[9/10] Skipped.")


# ── Step 10: GPU settings ──────────────────────────────────────────────────
if vram_gb >= 20:
    GPU_SETS = ["per_device_batch_size=16", "grad_accum_steps=1", "freeze_word_embeddings=false"]
elif vram_gb >= 12:
    GPU_SETS = ["per_device_batch_size=8", "grad_accum_steps=2", "freeze_word_embeddings=false"]
else:
    GPU_SETS = ["per_device_batch_size=4", "grad_accum_steps=4", "freeze_word_embeddings=true"]
print(f"\n[10/10] GPU: {vram_gb:.1f} GB -> {GPU_SETS}")


# ── Smoke training run ──────────────────────────────────────────────────────
SMOKE_RUN = "smoke_deberta_mt_s1"
print(f"\n{'='*70}")
print("  SMOKE TRAINING RUN (quick pipeline check)")
print(f"{'='*70}")
if not (ART_DIR / "runs" / SMOKE_RUN / "best" / "calibration.json").exists():
    if not run_finished(SMOKE_RUN):
        sh("ml/scripts/train_detector.py", "--set",
           f"run_name={SMOKE_RUN}", "max_train_rows=3000", "max_val_rows=1000",
           "epochs=1", "checkpoint_every_steps=0", *GPU_SETS)
    sh("ml/scripts/calibrate_thresholds.py", "--run", SMOKE_RUN)
    sh("ml/scripts/evaluate_detector.py", "--run", SMOKE_RUN, "--max-rows", "1000")
else:
    print("  Already done, skipping.")


# ── Full training run ───────────────────────────────────────────────────────
RUN_NAME = "deberta_multitask_s1"
print(f"\n{'='*70}")
print("  FULL TRAINING RUN (3 epochs with early stopping)")
print(f"{'='*70}")
if not run_finished(RUN_NAME):
    sh("ml/scripts/train_detector.py", "--set", f"run_name={RUN_NAME}", *GPU_SETS)
if not (ART_DIR / "runs" / RUN_NAME / "best" / "calibration.json").exists():
    sh("ml/scripts/calibrate_thresholds.py", "--run", RUN_NAME)
if not (ART_DIR / "runs" / RUN_NAME / "eval" / "metrics_summary.csv").exists():
    sh("ml/scripts/evaluate_detector.py", "--run", RUN_NAME)


# ── Done ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Model saved to: {ART_DIR / 'runs' / RUN_NAME / 'best'}")
print(f"  Restart demo_app.py --web and it will auto-detect the trained model.")
print(f"{'='*70}")

