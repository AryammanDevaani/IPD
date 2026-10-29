"""SemEval-2024 Task 8 Subtask A monolingual -> canonical rows (processed/semeval_a.parquet).

Publisher splits are kept and pinned: train -> train, dev -> validation, gold test -> test.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.data.io import PREPROCESSING_VERSION, rows_to_frame, write_table  # noqa: E402
from scriptsense_ml.data.sources import iter_semeval_a, semeval_a_row  # noqa: E402
from scriptsense_ml.paths import MANIFEST_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

if __name__ == "__main__":
    zip_path = RAW_DIR / "semeval2024_task8" / "SemEval2024-Task8-data.zip"
    rows, empty = [], 0
    for split, rec in iter_semeval_a(zip_path):
        if not str(rec.get("text", "")).strip():
            empty += 1
            continue
        rows.append(semeval_a_row(split, rec))
    df = rows_to_frame(rows)
    out = PROCESSED_DIR / "semeval_a.parquet"
    sha = write_table(df, out)
    record = {"source": "semeval_a", "rows": len(df), "dropped_empty_text": empty,
              "rows_by_split": dict(Counter(df["split"])), "output": out.name, "sha256": sha,
              "preprocessing_version": PREPROCESSING_VERSION}
    (MANIFEST_DIR / "prepared_semeval_a.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
