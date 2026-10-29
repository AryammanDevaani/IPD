"""Readers that turn each raw dataset into canonical ScriptSense rows (see schema.COLUMNS)."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import label_mapping as lm
from .normalization import decode_mixed_lines, dedup_key, normalize_for_model, stable_id

TXD22_LICENSE = "CC BY 4.0 - Iqbal & Kashem, TXD-22, Mendeley Data v1, doi:10.17632/prcjcggtjf.1"
SEMEVAL_LICENSE = "SemEval-2024 Task 8 (Zenodo 11002158); license unstated in record - research use only"
RAID_LICENSE = "RAID (Dugan et al., ACL 2024), HF liamdugan/raid, MIT per dataset card"


def _row(**kw) -> dict:
    raw = kw.pop("raw_text")
    base = {
        "row_id": f"{kw['source_dataset']}:{kw['source_row_id']}",
        "raw_text": raw,
        "text": normalize_for_model(raw),
        "original_document_id": "",
        "prompt_id": "",
        "generator": "",
        "domain": "",
        "attack_or_edit_type": "none",
        "split": "",
        "split_fixed": False,
        "group_keys": "",
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- TXD-22
def read_txd22(path: Path) -> tuple[list[dict], dict]:
    text, decode_stats = decode_mixed_lines(Path(path).read_bytes())
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    expected = ["Question", "Answer", "Source"]
    if list(df.columns) != expected:
        raise ValueError(f"TXD-22 schema changed: {list(df.columns)} != {expected}")
    rows = []
    for idx, rec in enumerate(df.itertuples(index=False)):
        c4, b, note = lm.map_txd22(rec.Source)
        # Normalised question text defines the prompt group; all 22 answers to it share a split.
        prompt_id = stable_id("txd22", dedup_key(rec.Question))
        gen = lm.txd22_generator(rec.Source)
        rows.append(_row(
            source_dataset="txd22",
            source_row_id=str(idx),
            raw_text=rec.Answer,
            prompt_id=prompt_id,
            generator=gen,
            attack_or_edit_type=gen.removeprefix("mixed:") if gen.startswith("mixed:") else "none",
            provenance=note,
            source_label=rec.Source,
            class_4way=int(c4),
            label_binary=int(b),
            group_keys=f"txd22:prompt:{prompt_id}",
            license_reference=TXD22_LICENSE,
        ))
    return rows, {"decode_line_stats": decode_stats, "raw_rows": len(df)}


# --------------------------------------------------------------------------- SemEval
SEMEVAL_A_FILES = {
    # member-name suffix -> (publisher split, our split)
    "subtaskA_train_monolingual.jsonl": "train",
    "subtaskA_dev_monolingual.jsonl": "validation",
    "subtaskA_test_monolingual.jsonl": "test",  # gold-labelled official test set
}


def iter_semeval_a(zip_path: Path) -> Iterator[tuple[str, dict]]:
    """Yield (our_split, raw_json_record) for Subtask A monolingual files in the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = member.rsplit("/", 1)[-1]
            if name.startswith("._") or name not in SEMEVAL_A_FILES:
                continue
            split = SEMEVAL_A_FILES[name]
            with zf.open(member) as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    if line.strip():
                        yield split, json.loads(line)


def semeval_a_row(split: str, rec: dict) -> dict:
    c4, b, note = lm.map_semeval_a(int(rec["label"]))
    rid = f"{split}-{rec['id']}"
    return _row(
        source_dataset="semeval_a",
        source_row_id=rid,
        raw_text=rec["text"],
        generator=str(rec.get("model", "")),
        domain=str(rec.get("source", "")),
        provenance=note,
        source_label=str(rec["label"]),
        class_4way=int(c4),
        label_binary=int(b),
        split=split,
        split_fixed=True,
        group_keys=f"semeval_a:doc:{rid}",
        license_reference=SEMEVAL_LICENSE,
    )


# --------------------------------------------------------------------------- RAID
def raid_row(rec: dict, line_no: int) -> dict | None:
    c4, b, note = lm.map_raid(rec["model"], rec["attack"])
    if c4 == lm.UNKNOWN and b == lm.UNKNOWN:
        return None
    lineage = rec["source_id"]
    return _row(
        source_dataset="raid",
        source_row_id=str(rec["id"]) or f"line{line_no}",
        raw_text=rec["generation"],
        original_document_id=lineage,
        prompt_id=stable_id("raid", rec.get("title", "")),
        generator=str(rec["model"]),
        domain=str(rec["domain"]),
        attack_or_edit_type=str(rec["attack"]),
        provenance=note + f"; decoding={rec.get('decoding') or 'n/a'}, rep_penalty={rec.get('repetition_penalty') or 'n/a'}",
        source_label=f"model={rec['model']};attack={rec['attack']}",
        class_4way=int(c4),
        label_binary=int(b),
        group_keys=f"raid:lineage:{lineage}",
        license_reference=RAID_LICENSE,
    )
