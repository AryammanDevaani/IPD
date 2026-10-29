"""Profile raw datasets before any training: schema, counts, labels, lengths, missing values,
model/domain/attack distribution and license. Writes data/manifests/raw_profile_<source>.json.

Usage:
    python ml/scripts/inspect_datasets.py txd22 semeval raid
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.data import sources  # noqa: E402
from scriptsense_ml.data.schema import CLASS4_NAMES, UNKNOWN  # noqa: E402
from scriptsense_ml.paths import MANIFEST_DIR, RAW_DIR  # noqa: E402

TXD22_PATH = RAW_DIR / "txd22" / "TXD-22 A Large-Scale Benchmark Dataset.csv"
SEMEVAL_ZIP = RAW_DIR / "semeval2024_task8" / "SemEval2024-Task8-data.zip"
RAID_TRAIN = RAW_DIR / "raid" / "train.csv"


def _length_stats(words: pd.Series) -> dict:
    q = words.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).round(1)
    return {"mean": round(float(words.mean()), 1), **{f"p{int(k * 100)}": float(v) for k, v in q.items()}}


def _class_name(c: int) -> str:
    return "UNKNOWN" if c == UNKNOWN else CLASS4_NAMES[c]


def profile_rows(rows: list[dict], extra: dict) -> dict:
    df = pd.DataFrame(rows)
    df["words"] = df["text"].str.split().str.len()
    df["class_name"] = df["class_4way"].map(_class_name)
    return {
        **extra,
        "rows": len(df),
        "columns": list(df.columns),
        "empty_text_rows": int((df["text"].str.strip() == "").sum()),
        "rows_by_split": df["split"].replace("", "unassigned").value_counts().to_dict(),
        "source_label_counts": df["source_label"].value_counts().to_dict(),
        "class_4way_counts": df["class_name"].value_counts().to_dict(),
        "binary_counts": df["label_binary"].map({0: "HUMAN_ONLY", 1: "AI_INVOLVED", UNKNOWN: "UNKNOWN"})
                          .value_counts().to_dict(),
        "class_by_split": pd.crosstab(df["split"].replace("", "unassigned"), df["class_name"]).to_dict(),
        "generator_counts": df["generator"].value_counts().to_dict(),
        "domain_counts": df["domain"].replace("", "unknown").value_counts().to_dict(),
        "attack_or_edit_counts": df["attack_or_edit_type"].value_counts().to_dict(),
        "word_length_overall": _length_stats(df["words"]),
        "median_words_by_class": df.groupby("class_name")["words"].median().to_dict(),
        "exact_duplicate_texts_within_source": int(df["text"].duplicated().sum()),
        "license": rows[0]["license_reference"] if rows else "",
    }


def inspect_txd22() -> dict:
    rows, meta = sources.read_txd22(TXD22_PATH)
    df = pd.DataFrame(rows)
    per_prompt = df.groupby("prompt_id").size()
    meta.update({
        "path": str(TXD22_PATH.relative_to(RAW_DIR)),
        "unique_prompts": int(df["prompt_id"].nunique()),
        "answers_per_prompt": per_prompt.describe().round(2).to_dict(),
    })
    return profile_rows(rows, meta)


def inspect_semeval() -> dict:
    with __import__("zipfile").ZipFile(SEMEVAL_ZIP) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
    rows = [sources.semeval_a_row(split, rec) for split, rec in sources.iter_semeval_a(SEMEVAL_ZIP)]
    if not rows:
        raise SystemExit(f"No Subtask A monolingual files found. Zip members: {members[:50]}")
    return profile_rows(rows, {"path": str(SEMEVAL_ZIP.relative_to(RAW_DIR)), "zip_members": members})


RAID_META_COLS = ["id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty",
                  "attack", "domain"]


def inspect_raid(chunksize: int = 200_000) -> dict:
    """Streaming profile; the text column is read only to measure length."""
    counts = {k: Counter() for k in ("model", "attack", "domain", "decoding", "repetition_penalty")}
    model_attack, missing = Counter(), Counter()
    word_samples, ids, n = [], set(), 0
    dup_ids = 0
    lineages = set()
    rng = np.random.default_rng(0)
    for chunk in pd.read_csv(RAID_TRAIN, chunksize=chunksize, dtype=str, keep_default_na=False,
                             usecols=RAID_META_COLS + ["generation"]):
        n += len(chunk)
        for k in counts:
            counts[k].update(chunk[k].replace("", "<empty>"))
        model_attack.update(zip(chunk["model"], chunk["attack"]))
        for c in chunk.columns:
            missing[c] += int((chunk[c] == "").sum())
        for i in chunk["id"]:
            if i in ids:
                dup_ids += 1
            ids.add(i)
        lineages.update(chunk["source_id"])
        take = rng.random(len(chunk)) < 0.01
        word_samples.extend(chunk.loc[take, "generation"].str.split().str.len().tolist())
        print(f"  raid profiled {n:,} rows", flush=True)
    return {
        "path": str(RAID_TRAIN.relative_to(RAW_DIR)),
        "rows": n,
        "unique_ids": len(ids),
        "duplicate_ids": dup_ids,
        "unique_source_ids (lineages)": len(lineages),
        "empty_values_by_column": dict(missing),
        **{f"{k}_counts": dict(v) for k, v in counts.items()},
        "model_x_attack_counts": {f"{m}|{a}": c for (m, a), c in sorted(model_attack.items())},
        "word_length_1pct_sample": _length_stats(pd.Series(word_samples)),
        "license": sources.RAID_LICENSE,
    }


INSPECTORS = {"txd22": inspect_txd22, "semeval": inspect_semeval, "raid": inspect_raid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(INSPECTORS))
    args = ap.parse_args()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        print(f"== {name}", flush=True)
        prof = INSPECTORS[name]()
        out = MANIFEST_DIR / f"raw_profile_{name}.json"
        out.write_text(json.dumps(prof, indent=2, default=str))
        summary = {k: prof[k] for k in prof if k in ("rows", "class_4way_counts", "binary_counts",
                                                      "rows_by_split", "unique_prompts", "decode_line_stats",
                                                      "duplicate_ids", "attack_counts", "model_counts")}
        print(json.dumps(summary, indent=2, default=str))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
