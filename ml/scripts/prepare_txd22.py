"""TXD-22 -> canonical rows (processed/txd22.parquet). Splits are assigned later, grouped by prompt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.data.io import PREPROCESSING_VERSION, rows_to_frame, write_table  # noqa: E402
from scriptsense_ml.data.sources import read_txd22  # noqa: E402
from scriptsense_ml.paths import MANIFEST_DIR, PROCESSED_DIR, RAW_DIR  # noqa: E402

if __name__ == "__main__":
    rows, meta = read_txd22(RAW_DIR / "txd22" / "TXD-22 A Large-Scale Benchmark Dataset.csv")
    df = rows_to_frame(rows)
    out = PROCESSED_DIR / "txd22.parquet"
    sha = write_table(df, out)
    record = {"source": "txd22", "rows": len(df), "output": out.name, "sha256": sha,
              "preprocessing_version": PREPROCESSING_VERSION, **meta}
    (MANIFEST_DIR / "prepared_txd22.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
