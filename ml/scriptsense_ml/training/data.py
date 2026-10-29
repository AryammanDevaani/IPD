"""Training data selection: split filtering, ablation filters, documented subsets."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.schema import Class4

COLS = ["row_id", "text", "class_4way", "label_binary", "split", "source_dataset", "generator",
        "domain", "attack_or_edit_type", "prompt_id", "original_document_id"]


def load_dataset(path: Path, columns=COLS) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def apply_ablation(train: pd.DataFrame, ablation: dict) -> tuple[pd.DataFrame, dict]:
    """Ablations only ever remove rows from the TRAIN split, so every run is compared on the
    same validation/test data."""
    removed = {}
    keep = pd.Series(True, index=train.index)
    if ablation.get("exclude_raid_paraphrase"):
        m = (train.source_dataset == "raid") & (train.attack_or_edit_type == "paraphrase")
        removed["raid_paraphrase"] = int(m.sum())
        keep &= ~m
    if ablation.get("exclude_txd22_mixed"):
        m = (train.source_dataset == "txd22") & train.class_4way.isin(
            [Class4.AI_PARAPHRASED_OR_REFINED, Class4.MIXED_OR_AI_ASSISTED])
        removed["txd22_mixed_or_refined"] = int(m.sum())
        keep &= ~m
    for src in ablation.get("exclude_sources", []) or []:
        m = train.source_dataset == src
        removed[f"source:{src}"] = int(m.sum())
        keep &= ~m
    return train[keep], removed


def stratified_subset(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Deterministic subset keeping the (source, class) mix proportional. ``None`` = all rows."""
    if n is None or n >= len(df):
        return df
    strata = df.source_dataset + "/" + df.class_4way.astype(str) + "/" + df.label_binary.astype(str)
    frac = n / len(df)
    parts = [g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby(strata)]
    return pd.concat(parts).sort_values("row_id")


def row_ids_digest(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(df.row_id)).encode()).hexdigest()


class TextDataset:
    def __init__(self, df: pd.DataFrame):
        self.texts = df.text.tolist()
        self.y4 = df.class_4way.to_numpy(np.int64)
        self.y2 = df.label_binary.to_numpy(np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.y4[i], self.y2[i]
