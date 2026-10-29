"""Merge prepared sources, deduplicate, assign lineage-grouped splits, audit leakage.

Policy (also recorded in the report):
  1. Exact duplicates of the *model text* (normalize_for_model output) are collapsed to one
     row when labels agree; when labels conflict every copy is dropped as label noise.
  2. Rows with the same aggressive dedup key (case/punctuation-insensitive) or a MinHash
     near-duplicate (estimated Jaccard >= threshold) are always placed in one split group.
  3. Cross-source near-duplicates with agreeing labels are collapsed to one row; with
     conflicting labels all copies are dropped. Within-source near-duplicates are kept
     (e.g. a RAID generation and its perturbed variants are intentionally distinct samples).
  4. Components inherit publisher-fixed splits; an evaluation pin beats a train pin, so
     official validation/test material is never trained on. Components pinned to two
     different evaluation splits are dropped and reported.
  5. The script fails if any group key spans more than one split.

Outputs: processed/dataset.parquet, data/manifests/dataset_report.json, data/manifests/splits.csv.gz
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.data.deduplication import MinHasher, UnionFind, near_duplicate_pairs  # noqa: E402
from scriptsense_ml.data.io import PREPROCESSING_VERSION, file_sha256, write_table  # noqa: E402
from scriptsense_ml.data.normalization import text_hash, unique_trigram_ratio  # noqa: E402
from scriptsense_ml.data.schema import CLASS4_NAMES, UNKNOWN  # noqa: E402
from scriptsense_ml.data.splitting import assert_no_leakage, assign_splits, connected_components, leakage_report  # noqa: E402
from scriptsense_ml.paths import MANIFEST_DIR, ML_ROOT, PROCESSED_DIR  # noqa: E402

SOURCES = ["semeval_a", "raid", "txd22", "academic"]
_HASHER: MinHasher | None = None


def _init(num_perm: int, seed: int) -> None:
    global _HASHER
    _HASHER = MinHasher(num_perm=num_perm, seed=seed)


def _sigs(texts: list[str]) -> np.ndarray:
    return np.stack([_HASHER.signature(t) for t in texts])


def minhash_all(texts: list[str], num_perm: int, seed: int, workers: int | None = None) -> np.ndarray:
    workers = workers or max(1, min(12, os.cpu_count() or 2))  # Colab has only 2 CPUs
    batches = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(num_perm, seed)) as ex:
        return np.concatenate(list(ex.map(_sigs, batches)))


def label_key(df: pd.DataFrame) -> pd.Series:
    return df["class_4way"].astype(str) + "/" + df["label_binary"].astype(str)


def distribution(df: pd.DataFrame) -> dict:
    cname = df["class_4way"].map(lambda c: "UNKNOWN" if c == UNKNOWN else CLASS4_NAMES[c])
    return {
        "rows_by_split": df["split"].value_counts().to_dict(),
        "rows_by_source_split": {f"{s}/{p}": int(n) for (s, p), n in df.groupby(["source_dataset", "split"]).size().items()},
        "class_4way_by_split": pd.crosstab(cname, df["split"]).to_dict(),
        "binary_by_split": pd.crosstab(df["label_binary"], df["split"]).to_dict(),
        "class_4way_by_source_split": {f"{s}/{p}": g.value_counts().to_dict()
                                       for (s, p), g in cname.groupby([df["source_dataset"], df["split"]])},
        "domain_by_split": pd.crosstab(df["domain"].replace("", "unknown"), df["split"]).to_dict(),
        "generator_by_split": pd.crosstab(df["generator"].replace("", "unknown"), df["split"]).to_dict(),
        "attack_by_split": pd.crosstab(df["attack_or_edit_type"], df["split"]).to_dict(),
        "median_words_by_class_split": {f"{c}/{s}": float(v) for (c, s), v in
                                        df["text"].str.split().str.len().groupby([cname, df["split"]]).median().items()},
    }


def main() -> None:
    t0 = time.time()
    cfg = yaml.safe_load((ML_ROOT / "configs" / "data.yaml").read_text())
    dcfg, seed = cfg["dedup"], cfg["seed"]

    frames, inputs = [], {}
    for name in SOURCES:
        p = PROCESSED_DIR / f"{name}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
            inputs[name] = {"file": p.name, "sha256": file_sha256(p), "rows": len(frames[-1])}
    missing = [s for s in SOURCES if s not in inputs]
    df = pd.concat(frames, ignore_index=True)
    report: dict = {"created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
                    "preprocessing_version": PREPROCESSING_VERSION, "inputs": inputs,
                    "sources_missing": missing, "rows_before": len(df),
                    "rows_before_by_source_split": {f"{s}/{p or 'unassigned'}": int(n) for (s, p), n in
                                                    df.groupby(["source_dataset", "split"]).size().items()}}
    print(f"loaded {len(df):,} rows from {list(inputs)}", flush=True)
    dropped = Counter()

    # ---- 0. quality filter: degenerate repetition loops
    rep = df["text"].map(unique_trigram_ratio)
    degenerate = rep < cfg["quality"]["min_unique_trigram_ratio"]
    dropped["degenerate_repetition"] = int(degenerate.sum())
    report["degenerate_repetition_by_source_class"] = {
        f"{s}/{c}": int(n) for (s, c), n in df[degenerate].groupby(["source_dataset", "class_4way"]).size().items()}
    df = df[~degenerate].reset_index(drop=True)

    # ---- 1. exact duplicates of model text
    df["model_hash"] = df["text"].map(lambda t: __import__("hashlib").sha1(t.encode()).hexdigest())
    lk = label_key(df)
    conflict_hashes = set(df.assign(lk=lk).groupby("model_hash")["lk"].nunique().loc[lambda s: s > 1].index)
    is_conflict = df["model_hash"].isin(conflict_hashes)
    dropped["exact_dup_conflicting_labels"] = int(is_conflict.sum())
    df = df[~is_conflict]
    # Prefer keeping evaluation-pinned copies so the split pin survives the collapse.
    pri = df["split"].map({"test": 0, "challenge": 1, "validation": 2}).fillna(3)
    df = df.assign(_pri=pri).sort_values(["_pri", "row_id"])
    dup_mask = df.duplicated("model_hash", keep="first")
    # Keep the dropped copies' group keys on the survivor so their lineage still constrains splits.
    extra_keys = df[dup_mask].groupby("model_hash")["group_keys"].agg("|".join)
    dropped["exact_dup_same_labels"] = int(dup_mask.sum())
    df = df[~dup_mask].drop(columns="_pri").reset_index(drop=True)
    df["group_keys"] = [g + ("|" + extra_keys[h] if h in extra_keys.index else "")
                        for g, h in zip(df["group_keys"], df["model_hash"])]
    print(f"exact dedup: {dict(dropped)} -> {len(df):,}", flush=True)

    # ---- 2. aggressive-key groups + MinHash near-duplicates
    df["dedup_hash"] = df["text"].map(text_hash)
    sigs = minhash_all(df["text"].tolist(), dcfg["num_perm"], seed)
    print(f"minhash done {time.time() - t0:.0f}s", flush=True)
    pairs = near_duplicate_pairs(sigs, bands=dcfg["bands"], threshold=dcfg["near_dup_threshold"])
    del sigs
    src = df["source_dataset"].to_numpy()
    labels = label_key(df).to_numpy()
    uf = UnionFind(len(df))
    cross_conflict, cross_same, pair_stats = set(), [], Counter()
    for i, j, _ in pairs:
        uf.union(i, j)
        cross = src[i] != src[j]
        same = labels[i] == labels[j]
        pair_stats[f"{'cross' if cross else 'within'}_source/{'same' if same else 'diff'}_labels"] += 1
        if cross and not same:
            cross_conflict.update((i, j))
        elif cross and same:
            cross_same.append((i, j))
    report["near_duplicate_pairs"] = dict(pair_stats)
    near_root = [uf.find(i) for i in range(len(df))]
    df["group_keys"] = [f"{g}|dup:exact:{h}|dup:near:{df.at[r, 'row_id']}" for g, h, r in
                        zip(df["group_keys"], df["dedup_hash"], near_root)]

    drop_idx = set(cross_conflict)
    dropped["cross_source_near_dup_conflicting_labels"] = len(cross_conflict)
    # collapse cross-source same-label near-dups: keep the higher-priority copy of each pair
    pri = df["split"].map({"test": 0, "challenge": 1, "validation": 2}).fillna(3).to_numpy()
    n_same = 0
    for i, j in cross_same:
        if i in drop_idx or j in drop_idx:
            continue
        loser = j if (pri[i], df.at[i, "row_id"]) <= (pri[j], df.at[j, "row_id"]) else i
        drop_idx.add(loser)
        n_same += 1
    dropped["cross_source_near_dup_same_labels"] = n_same
    keep = ~df.index.isin(sorted(drop_idx))
    df = df[keep].reset_index(drop=True)
    print(f"near-dup: pairs={dict(pair_stats)} dropped={dict(dropped)} -> {len(df):,}", flush=True)

    # ---- 3. components and split assignment
    keys = [k.split("|") for k in df["group_keys"]]
    comps = connected_components(keys)
    row_ids = df["row_id"].tolist()
    comp_name: dict[int, str] = {}
    for c, rid in zip(comps, row_ids):
        comp_name[c] = min(comp_name.get(c, rid), rid)
    fixed = [s if f else "" for s, f in zip(df["split"], df["split_fixed"])]
    splits, conflicts = assign_splits(comps, comp_name, fixed, cfg["txd22"]["ratios"], seed)
    moved = sum(1 for f, s in zip(fixed, splits) if f and s and f != s)
    audit = df.assign(new_split=splits, component=comps, fixed_split=fixed)
    audit = audit[(audit.new_split == "") | ((audit.fixed_split != "") & (audit.fixed_split != audit.new_split))]
    comp_sizes = Counter(comps)
    audit = audit.assign(component_size=audit.component.map(comp_sizes),
                         words=audit.text.str.split().str.len(), text_head=audit.text.str[:120])
    audit[["row_id", "source_dataset", "generator", "domain", "attack_or_edit_type", "class_4way", "fixed_split",
           "new_split", "component", "component_size", "words", "text_head"]].to_csv(
        MANIFEST_DIR / "split_moves_and_conflicts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    report["largest_components"] = [{"size": n, "splits_pinned": sorted(set(f for f, c in zip(fixed, comps) if c == comp and f))}
                                    for comp, n in comp_sizes.most_common(5)]
    df["split"] = splits
    dropped["component_conflicting_eval_pins"] = int((df["split"] == "").sum())
    df = df[df["split"] != ""].reset_index(drop=True)
    report["pinned_rows_moved_to_eval_split"] = moved
    report["conflicting_components"] = len(conflicts)
    report["components"] = len(set(comps))

    # ---- 4. leakage audit (hard failure)
    keys = [k.split("|") for k in df["group_keys"]]
    report["leakage"] = leakage_report(df["split"].tolist(), keys)
    assert_no_leakage(df["split"].tolist(), keys)
    for col_a, col_b in (("dedup_hash", "split"), ("model_hash", "split")):
        n = int((df.groupby(col_a)[col_b].nunique() > 1).sum())
        report["leakage"][f"{col_a}_spanning_splits"] = n
        assert n == 0, f"{col_a} spans splits"

    # ---- 5. outputs
    report["dropped"] = dict(dropped)
    report["rows_after"] = len(df)
    report["distribution_after"] = distribution(df)
    report["train_rows_after_dedup"] = int((df["split"] == "train").sum())
    report["train_rows_target_before_dedup"] = 284557
    report["notes"] = [
        "Counts are actual post-filter counts; the 284,557 figure is the planning target only.",
        "academic source is absent unless processed/academic.parquet exists (requires consented human data).",
        "RAID human_clean training rows are an addition to the CLAUDE.md budget (domain-shortcut mitigation); see ml/configs/data.yaml.",
    ]
    out = PROCESSED_DIR / "dataset.parquet"
    report["output"] = {"file": out.name, "sha256": write_table(df.drop(columns=["model_hash"]), out)}
    df[["row_id", "source_dataset", "split", "class_4way", "label_binary"]].to_csv(
        MANIFEST_DIR / "splits.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    report["split_manifest"] = {"file": "splits.csv.gz", "sha256": file_sha256(MANIFEST_DIR / "splits.csv.gz")}
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (MANIFEST_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("rows_before", "dropped", "rows_after", "train_rows_after_dedup",
                                             "pinned_rows_moved_to_eval_split", "conflicting_components")}, indent=2))
    print(json.dumps(report["distribution_after"]["rows_by_source_split"], indent=2))


if __name__ == "__main__":
    main()
