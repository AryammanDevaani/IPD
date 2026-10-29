"""Read/write canonical row tables with schema validation and content hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .schema import COLUMNS, validate_row

PREPROCESSING_VERSION = "1.0.0"


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    for r in rows:
        validate_row(r)
    df = pd.DataFrame(rows, columns=COLUMNS)
    dup = df["row_id"].duplicated()
    if dup.any():
        raise ValueError(f"duplicate row_id values, e.g. {df.loc[dup, 'row_id'].head(3).tolist()}")
    return df


def write_table(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return file_sha256(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()
