"""Train the AI-writing detector on processed/dataset.parquet.

Usage:
    python ml/scripts/train_detector.py --config ml/configs/ai_detector.yaml [--set key=value ...]
e.g. --set model_name=roberta-base --set mode=binary --set run_name=roberta_binary_s1
     --set max_train_rows=2000 --set max_val_rows=500   (documented smoke subset)

Refuses to start if any normalized text appears in both train and an evaluation split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.data.io import file_sha256  # noqa: E402
from scriptsense_ml.data.normalization import text_hash  # noqa: E402
from scriptsense_ml.paths import ARTIFACT_DIR, MANIFEST_DIR, PROCESSED_DIR  # noqa: E402
from scriptsense_ml.training.data import (  # noqa: E402
    apply_ablation, load_dataset, row_ids_digest, stratified_subset,
)
from scriptsense_ml.training.trainer import train  # noqa: E402


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def check_text_leakage(df, ds_path: Path, ds_sha: str) -> None:
    """Fail if any normalized text is in train and an evaluation split. The (slow) result is cached
    next to the dataset, keyed by its SHA-256, so a changed dataset is always re-checked."""
    cache = ds_path.with_name(ds_path.name + ".leakage_ok.json")
    if cache.exists() and json.loads(cache.read_text()).get("sha256") == ds_sha:
        print("leakage check: cached pass for this exact dataset", flush=True)
        return
    h = df.text.map(text_hash)
    train_h = set(h[df.split == "train"])
    for s in sorted(set(df.split) - {"train"}):
        n = int(h[df.split == s].isin(train_h).sum())
        if n:
            raise SystemExit(f"LEAKAGE: {n} {s} texts also occur in train; rebuild the dataset")
    cache.write_text(json.dumps({"sha256": ds_sha, "result": "no train/eval text overlap"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "ai_detector.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--dataset", default=str(PROCESSED_DIR / "dataset.parquet"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.update(parse_overrides(args.set))
    ds_path = Path(args.dataset)
    df = load_dataset(ds_path)
    ds_sha = file_sha256(ds_path)
    check_text_leakage(df, ds_path, ds_sha)

    train_df = df[df.split == "train"]
    val_df = df[df.split == "validation"]
    train_df, removed = apply_ablation(train_df, cfg.get("ablation") or {})
    train_df = stratified_subset(train_df, cfg.get("max_train_rows"), cfg["seed"])
    val_df = stratified_subset(val_df, cfg.get("max_val_rows"), cfg["seed"])

    report_path = MANIFEST_DIR / "dataset_report.json"
    provenance = {
        "dataset_file": ds_path.name,
        "dataset_sha256": ds_sha,
        "dataset_report_sha256": file_sha256(report_path) if report_path.exists() else None,
        "train_rows": len(train_df), "validation_rows": len(val_df),
        "train_row_ids_sha256": row_ids_digest(train_df), "validation_row_ids_sha256": row_ids_digest(val_df),
        "ablation_removed_rows": removed,
        "is_subset": cfg.get("max_train_rows") is not None or cfg.get("max_val_rows") is not None,
        "train_rows_by_source": train_df.source_dataset.value_counts().to_dict(),
    }
    run_dir = ARTIFACT_DIR / "runs" / cfg["run_name"]
    print(json.dumps({"run_dir": str(run_dir), **provenance}, indent=2), flush=True)
    meta = train(cfg, train_df, val_df, run_dir, provenance)
    if "interrupted_at_step" in meta:
        print(f"Stopped at step {meta['interrupted_at_step']}; re-run the same command to resume.")
    else:
        print(json.dumps({"best": meta["best"], "steps": meta["optimizer_steps_done"]}, indent=2))


if __name__ == "__main__":
    main()
