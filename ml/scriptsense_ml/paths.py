"""Filesystem locations for ScriptSense ML data.

Raw and processed datasets are large and/or licensed, so they live outside the
repository (which sits in a cloud-synced folder). Override with the
SCRIPTSENSE_DATA_DIR environment variable. Manifests are small and tracked.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"

DATA_DIR = Path(os.environ.get("SCRIPTSENSE_DATA_DIR", Path.home() / "scriptsense_data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = Path(os.environ.get("SCRIPTSENSE_MANIFEST_DIR", REPO_ROOT / "data" / "manifests"))
ARTIFACT_DIR = Path(os.environ.get("SCRIPTSENSE_ARTIFACT_DIR", DATA_DIR / "artifacts"))


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MANIFEST_DIR, ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)
