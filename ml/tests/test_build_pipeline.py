"""End-to-end smoke test of deduplicate_and_split on a tiny SYNTHETIC fixture.

The fixture is hand-written test data, isolated in a temp dir; it is never mixed into
real datasets or benchmark results.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scriptsense_ml.data.io import rows_to_frame

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "scripts"))
import prepare_raid  # noqa: E402

SYN = "SYNTHETIC TEST FIXTURE - not real data"
TOPICS = ["rivers", "volcanoes", "bridges", "orchards", "satellites", "glaciers", "markets", "libraries",
          "harbors", "deserts", "forests", "railways"]


def sentence(topic, k, variant=0):
    return (f"Paragraph {k} about {topic}: the regional survey of {topic} recorded measurement {k * 7 + variant} "
            f"and observers noted that {topic} in district {k} changed after season {k + 3}, while volunteers "
            f"catalogued {k * 11} distinct features near the {topic} station number {k * 13}.")


def row(src, rid, text, c4, b, split="", fixed=False, keys="", prompt=""):
    return dict(row_id=f"{src}:{rid}", text=text, raw_text=text, source_dataset=src, source_row_id=rid,
                original_document_id="", prompt_id=prompt, generator="synthetic", domain="synthetic",
                attack_or_edit_type="none", provenance=SYN, source_label=str(c4), class_4way=c4,
                label_binary=b, split=split, split_fixed=fixed, group_keys=keys or f"{src}:doc:{rid}",
                license_reference=SYN)


@pytest.fixture()
def fixture_dirs(tmp_path):
    data, manifests = tmp_path / "data", tmp_path / "manifests"
    (data / "processed").mkdir(parents=True)
    manifests.mkdir()
    sem = [row("semeval_a", f"tr{i}", sentence(TOPICS[i % 12], i), i % 2, i % 2, "train", True) for i in range(40)]
    sem += [row("semeval_a", f"va{i}", sentence(TOPICS[i % 12], 100 + i), i % 2, i % 2, "validation", True)
            for i in range(10)]
    sem += [row("semeval_a", f"te{i}", sentence(TOPICS[i % 12], 200 + i), i % 2, i % 2, "test", True)
            for i in range(10)]
    # exact duplicate of a test row placed in train -> must be removed from training
    sem.append(row("semeval_a", "trdup", sentence(TOPICS[0], 200), 0, 0, "train", True))
    # same text, conflicting label -> all copies dropped
    sem.append(row("semeval_a", "trconf", sentence(TOPICS[1], 201), 0, 0, "train", True))
    txd = []
    for p in range(30):
        for c in range(4):
            txd.append(row("txd22", f"{p}-{c}", sentence(TOPICS[c], 1000 + p * 4 + c, c), c, int(c > 0),
                           keys=f"txd22:prompt:{p}", prompt=str(p)))
    rows_to_frame(sem).to_parquet(data / "processed" / "semeval_a.parquet", index=False)
    rows_to_frame(txd).to_parquet(data / "processed" / "txd22.parquet", index=False)
    return data, manifests


def test_build_pipeline_end_to_end(fixture_dirs):
    data, manifests = fixture_dirs
    env = {**os.environ, "SCRIPTSENSE_DATA_DIR": str(data), "SCRIPTSENSE_MANIFEST_DIR": str(manifests)}
    res = subprocess.run([sys.executable, str(ML / "scripts" / "deduplicate_and_split.py")],
                         env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = pd.read_parquet(data / "processed" / "dataset.parquet")
    report = json.loads((manifests / "dataset_report.json").read_text())

    assert "semeval_a:trdup" not in set(out["row_id"])              # test copy kept, train copy collapsed
    assert out.loc[out.row_id == "semeval_a:te0", "split"].item() == "test"
    assert not out["row_id"].isin(["semeval_a:trconf", "semeval_a:te1"]).any()  # conflicting labels dropped
    assert report["dropped"]["exact_dup_conflicting_labels"] == 2
    # official splits preserved
    assert set(out.loc[out.row_id.str.startswith("semeval_a:va"), "split"]) == {"validation"}
    # every TXD prompt lives in exactly one split
    txd = out[out.source_dataset == "txd22"]
    assert (txd.groupby("prompt_id")["split"].nunique() == 1).all()
    assert report["leakage"]["num_leaking_keys"] == 0


def test_raid_category_and_water_fill():
    assert prepare_raid.category("human", "none") == "human_clean"
    assert prepare_raid.category("human", "paraphrase") is None
    assert prepare_raid.category("gpt4", "paraphrase") == "ai_paraphrase"
    assert prepare_raid.category("gpt4", "homoglyph") == "ai_other"
    alloc = prepare_raid.water_fill({"a": 2, "b": 100, "c": 100}, 50)
    assert alloc["a"] == 2 and alloc["b"] + alloc["c"] == 48 and sum(alloc.values()) == 50
    assert sum(prepare_raid.water_fill({"a": 3, "b": 4}, 50).values()) == 7  # never pads


def test_raid_holdout_generator_never_in_train_pool():
    cfg = {"seed": 1, "raid": {"challenge_lineage_fraction": 0.1, "validation_lineage_fraction": 0.05,
                               "holdout_generators": ["cohere"]}}

    class R:  # minimal row stub
        def __init__(self, model, sid):
            self.model, self.attack, self.domain, self.source_id = model, "none", "news", sid

    pools = [prepare_raid.classify(R("cohere", f"s{i}"), cfg) for i in range(500)]
    assert all(p is None or p[0] == "challenge" for p in pools)
    assert any(p is not None for p in pools)
