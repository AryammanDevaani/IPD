"""Stratified, lineage-safe RAID sample -> processed/raid.parquet.

Pass 1 streams only metadata columns and counts eligible rows per stratum.
Quotas are spread across strata by water-filling (equal share, leftovers
redistributed to strata that still have rows). Pass 2 streams again and keeps,
per stratum, the rows with the smallest seeded SHA-256 keys - deterministic and
independent of file order. Rows of a challenge lineage or a held-out generator
family can never enter the training pool.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.data import label_mapping as lm  # noqa: E402
from scriptsense_ml.data.normalization import normalize_for_model  # noqa: E402
from scriptsense_ml.data.io import PREPROCESSING_VERSION, rows_to_frame, write_table  # noqa: E402
from scriptsense_ml.data.sources import raid_row  # noqa: E402
from scriptsense_ml.paths import MANIFEST_DIR, ML_ROOT, PROCESSED_DIR, RAW_DIR  # noqa: E402

RAID_TRAIN = RAW_DIR / "raid" / "train.csv"
META = ["id", "source_id", "model", "attack", "domain"]
CHUNK = 200_000


def _u(seed: int, key: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64


def category(model: str, attack: str) -> str | None:
    human = model == "human"
    if attack == "none":
        return "human_clean" if human else "ai_clean"
    if human:
        return None  # human attacked rows are outside this budget
    if attack == "paraphrase":
        return "ai_paraphrase"
    return "ai_other"


def stratum(cat: str, model: str, attack: str, domain: str) -> str:
    if cat == "ai_other":
        return f"{model}|{attack}|{domain}"
    if cat == "human_clean":
        return domain
    return f"{model}|{domain}"


def classify(rec, cfg) -> tuple[str, str, str] | None:
    """Return (pool, category, stratum) or None if the row is ineligible."""
    cat = category(rec.model, rec.attack)
    if cat is None:
        return None
    u = _u(cfg["seed"], "lineage:" + rec.source_id)
    c_frac = cfg["raid"]["challenge_lineage_fraction"]
    holdout = rec.model in cfg["raid"]["holdout_generators"]
    if u < c_frac:
        pool = "challenge"
    elif holdout:
        return None  # held-out generator outside challenge lineages: usable by no pool
    elif u < c_frac + cfg["raid"]["validation_lineage_fraction"]:
        pool = "validation"
    else:
        pool = "train"
    return pool, cat, stratum(cat, rec.model, rec.attack, rec.domain)


def water_fill(available: dict[str, int], quota: int) -> dict[str, int]:
    alloc = {s: 0 for s in available}
    remaining, open_ = quota, {s for s, n in available.items() if n > 0}
    while remaining > 0 and open_:
        share = max(1, remaining // len(open_))
        for s in sorted(open_):
            take = min(share, available[s] - alloc[s], remaining)
            alloc[s] += take
            remaining -= take
            if alloc[s] >= available[s]:
                open_.discard(s)
            if remaining == 0:
                break
    return alloc


def main() -> None:
    cfg = yaml.safe_load((ML_ROOT / "configs" / "data.yaml").read_text())
    quotas = {"train": cfg["raid"]["train_quota"], "validation": cfg["raid"]["validation_quota"],
              "challenge": cfg["raid"]["challenge_quota"]}

    # ---- pass 1: counts
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    total = 0
    for chunk in pd.read_csv(RAID_TRAIN, usecols=META, dtype=str, keep_default_na=False, chunksize=CHUNK):
        for rec in chunk.itertuples(index=False):
            c = classify(rec, cfg)
            if c:
                counts[(c[0], c[1])][c[2]] += 1
        total += len(chunk)
        print(f"pass1 {total:,}", flush=True)

    alloc: dict[tuple[str, str], dict[str, int]] = {}
    for pool, q in quotas.items():
        for cat, n in q.items():
            alloc[(pool, cat)] = water_fill(dict(counts[(pool, cat)]), int(n))

    # ---- pass 2: keep smallest keys per stratum (max-heap of negated keys)
    heaps: dict[tuple[str, str, str], list] = defaultdict(list)
    empty_after_norm: Counter = Counter()
    seen = 0
    for chunk in pd.read_csv(RAID_TRAIN, dtype=str, keep_default_na=False, chunksize=CHUNK):
        for line_no, rec in zip(range(seen, seen + len(chunk)), chunk.itertuples(index=False)):
            c = classify(rec, cfg)
            if not c:
                continue
            pool, cat, st = c
            cap = alloc[(pool, cat)].get(st, 0)
            if cap == 0:
                continue
            key = _u(cfg["seed"], f"row:{rec.id}:{line_no}")
            h = heaps[(pool, cat, st)]
            if len(h) >= cap and -h[0][0] <= key:
                continue
            if not normalize_for_model(rec.generation):
                empty_after_norm[(pool, cat)] += 1  # e.g. only invisible characters
                continue
            item = (-key, line_no, rec._asdict())
            if len(h) < cap:
                heapq.heappush(h, item)
            elif -h[0][0] > key:
                heapq.heapreplace(h, item)
        seen += len(chunk)
        print(f"pass2 {seen:,}", flush=True)

    rows, skipped_empty, selected, used_ids = [], 0, Counter(), set()
    for (pool, cat, st), h in heaps.items():
        for _, line_no, rec in h:
            if not normalize_for_model(rec["generation"]):
                skipped_empty += 1
                continue
            row = raid_row(rec, line_no)
            if row is None:
                continue
            if row["row_id"] in used_ids:  # RAID ids are expected unique; disambiguate if not
                row["source_row_id"] = f"{rec['id']}@{line_no}"
                row["row_id"] = f"raid:{row['source_row_id']}"
            used_ids.add(row["row_id"])
            selected[(pool, cat)] += 1
            row["split"] = pool
            row["split_fixed"] = True
            row["attack_or_edit_type"] = rec["attack"]
            rows.append(row)
    rows.sort(key=lambda r: r["row_id"])
    df = rows_to_frame(rows)
    out = PROCESSED_DIR / "raid.parquet"
    sha = write_table(df, out)

    report = {
        "source": "raid", "raw_rows_scanned": total, "rows": len(df), "output": out.name, "sha256": sha,
        "preprocessing_version": PREPROCESSING_VERSION, "config": cfg["raid"], "seed": cfg["seed"],
        "skipped_empty_text": skipped_empty,
        "skipped_empty_after_normalization_during_selection": {f"{p}/{c}": n for (p, c), n in empty_after_norm.items()},
        "target_vs_actual": {
            f"{pool}/{cat}": {"target": int(n), "eligible": int(sum(counts[(pool, cat)].values())),
                              "selected": selected[(pool, cat)]}
            for pool, q in quotas.items() for cat, n in q.items()
        },
        "class_4way_by_split": pd.crosstab(df["split"], df["class_4way"]).to_dict(),
        "generator_by_split": pd.crosstab(df["generator"], df["split"]).to_dict(),
        "attack_by_split": pd.crosstab(df["attack_or_edit_type"], df["split"]).to_dict(),
        "domain_by_split": pd.crosstab(df["domain"], df["split"]).to_dict(),
        "strata_fully_used": {
            f"{p}/{c}": {s: n for s, n in counts[(p, c)].items() if alloc[(p, c)].get(s, 0) >= n}
            for (p, c) in alloc
        },
    }
    lineage_pools = df.groupby("original_document_id")["split"].nunique()
    report["lineages_in_both_pools"] = int((lineage_pools > 1).sum())
    if report["lineages_in_both_pools"]:
        raise SystemExit("RAID lineage leakage between pools")
    (MANIFEST_DIR / "prepared_raid.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("rows", "target_vs_actual", "lineages_in_both_pools")}, indent=2))


if __name__ == "__main__":
    main()
