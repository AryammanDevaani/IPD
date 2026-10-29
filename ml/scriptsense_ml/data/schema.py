"""Canonical row schema and label taxonomy shared by data, training, inference and API."""
from __future__ import annotations

from enum import IntEnum

UNKNOWN = -1  # label not supported by the source's provenance; excluded from that head's loss


class Class4(IntEnum):
    HUMAN_ONLY = 0
    AI_GENERATED = 1
    AI_PARAPHRASED_OR_REFINED = 2
    MIXED_OR_AI_ASSISTED = 3


CLASS4_NAMES = [c.name for c in Class4]
BINARY_NAMES = ["HUMAN_ONLY", "AI_INVOLVED"]

SPLITS = ("train", "validation", "test", "challenge")

# Columns every processed row must carry (CLAUDE.md section 4, data governance).
COLUMNS = [
    "row_id",               # globally unique: f"{source_dataset}:{source_row_id}"
    "text",                 # normalize_for_model(raw_text)
    "raw_text",
    "source_dataset",
    "source_row_id",
    "original_document_id",  # lineage root if known (e.g. RAID source_id)
    "prompt_id",
    "generator",
    "domain",
    "attack_or_edit_type",
    "provenance",           # short human-readable explanation of why the label holds
    "source_label",         # the dataset's original label, verbatim
    "class_4way",           # Class4 value or UNKNOWN
    "label_binary",         # 0/1 or UNKNOWN
    "split",                # fixed by the source (official dev/test) or "" until assigned
    "split_fixed",          # True when the split comes from the dataset publisher
    "group_keys",           # "|"-joined lineage keys that must not cross splits
    "license_reference",
]


def binary_from_class4(c4: int) -> int:
    if c4 == UNKNOWN:
        return UNKNOWN
    return 0 if c4 == Class4.HUMAN_ONLY else 1


def validate_row(row: dict) -> None:
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise ValueError(f"row {row.get('row_id')} missing columns {missing}")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError(f"row {row['row_id']} has empty text")
    c4, b = row["class_4way"], row["label_binary"]
    if c4 not in (UNKNOWN, *iter(Class4)):
        raise ValueError(f"row {row['row_id']} bad class_4way {c4}")
    if b not in (UNKNOWN, 0, 1):
        raise ValueError(f"row {row['row_id']} bad label_binary {b}")
    if c4 != UNKNOWN and b != UNKNOWN and binary_from_class4(c4) != b:
        raise ValueError(f"row {row['row_id']} binary {b} inconsistent with class {c4}")
    if c4 == UNKNOWN and b == UNKNOWN:
        raise ValueError(f"row {row['row_id']} carries no usable label")
