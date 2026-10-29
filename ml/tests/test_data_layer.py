import numpy as np
import pytest

from scriptsense_ml.data import label_mapping as lm
from scriptsense_ml.data.deduplication import MinHasher, exact_duplicate_groups, near_duplicate_pairs
from scriptsense_ml.data.normalization import (
    decode_mixed_lines, dedup_key, normalize_for_model, text_hash,
)
from scriptsense_ml.data.schema import UNKNOWN, Class4, binary_from_class4, validate_row
from scriptsense_ml.data.splitting import (
    LeakageError, assert_no_leakage, assign_splits, connected_components, leakage_report,
)


# ---------------------------------------------------------------- normalization
def test_normalize_for_model_strips_invisible_but_keeps_case_and_paragraphs():
    raw = "Hello​  World!\r\n\r\n\r\n  Next para﻿"
    assert normalize_for_model(raw) == "Hello World!\n\nNext para"


def test_dedup_key_is_case_punct_and_homoglyph_width_insensitive():
    assert dedup_key("Ｈello, WORLD!!") == dedup_key("hello world")
    assert text_hash("A  b.c") == text_hash("a b c")


def test_decode_mixed_lines_handles_utf8_and_cp1252():
    raw = "café utf8\n".encode("utf-8") + b"it\x92s cp1252\n" + b"\xef\xbb\xbfx"
    text, stats = decode_mixed_lines(b"\xef\xbb\xbf" + raw)
    assert text.splitlines()[:2] == ["café utf8", "it’s cp1252"]
    assert stats["cp1252"] == 1 and stats["latin1_fallback"] == 0


# ---------------------------------------------------------------- labels
def test_binary_from_class4():
    assert binary_from_class4(Class4.HUMAN_ONLY) == 0
    for c in (Class4.AI_GENERATED, Class4.AI_PARAPHRASED_OR_REFINED, Class4.MIXED_OR_AI_ASSISTED):
        assert binary_from_class4(c) == 1
    assert binary_from_class4(UNKNOWN) == UNKNOWN


def test_semeval_mapping():
    assert lm.map_semeval_a(0)[:2] == (Class4.HUMAN_ONLY, 0)
    assert lm.map_semeval_a(1)[:2] == (Class4.AI_GENERATED, 1)
    with pytest.raises(ValueError):
        lm.map_semeval_a(2)


@pytest.mark.parametrize("model,attack,expected", [
    ("human", "none", (Class4.HUMAN_ONLY, 0)),
    ("gpt4", "none", (Class4.AI_GENERATED, 1)),
    ("gpt4", "paraphrase", (Class4.AI_PARAPHRASED_OR_REFINED, 1)),
    ("human", "paraphrase", (Class4.MIXED_OR_AI_ASSISTED, 1)),
    # character-level perturbations are never "humanized"
    ("mistral", "homoglyph", (Class4.AI_GENERATED, 1)),
    ("mistral", "zero_width_space", (Class4.AI_GENERATED, 1)),
    ("human", "whitespace", (Class4.HUMAN_ONLY, 0)),
    # synonym swaps: AI involvement known, 4-way label unsupported
    ("chatgpt", "synonym", (UNKNOWN, 1)),
    ("human", "synonym", (UNKNOWN, UNKNOWN)),
])
def test_raid_mapping(model, attack, expected):
    assert lm.map_raid(model, attack)[:2] == expected


def test_raid_unknown_attack_raises():
    with pytest.raises(ValueError):
        lm.map_raid("gpt4", "new_attack")


def test_txd22_mapping_covers_all_22_published_labels():
    published = [
        "ChatGPT", "Grok", "Copilot", "Gemini", "DeepSeek", "DeepAI", "Qwen", "Claude", "Llama",
        "Gemma", "Z.ai", "Perplexity", "Mistral AI", "Poe", "Nova", "BLACKBOXAI", "Pi",
        "Human written",
        "AI-generated and Human-written",  # the raw file uses a non-breaking space here
        "AI-generated and AI-refined",
        "Human-written and AI-refined",
        "AI-generated and AI-refined and Human-written",
    ]
    got = {s: lm.map_txd22(s)[0] for s in published}
    assert sum(v == Class4.AI_GENERATED for v in got.values()) == 17
    assert got["Human written"] == Class4.HUMAN_ONLY
    assert got["AI-generated and AI-refined"] == Class4.AI_PARAPHRASED_OR_REFINED
    assert got["AI-generated and Human-written"] == Class4.MIXED_OR_AI_ASSISTED
    assert got["Human-written and AI-refined"] == Class4.MIXED_OR_AI_ASSISTED
    assert got["AI-generated and AI-refined and Human-written"] == Class4.MIXED_OR_AI_ASSISTED
    with pytest.raises(ValueError):
        lm.map_txd22("Some new model")


def _valid_row(**kw):
    row = dict(row_id="x:1", text="hello", raw_text="hello", source_dataset="x", source_row_id="1",
               original_document_id="", prompt_id="", generator="", domain="", attack_or_edit_type="none",
               provenance="", source_label="", class_4way=0, label_binary=0, split="", split_fixed=False,
               group_keys="", license_reference="")
    row.update(kw)
    return row


def test_validate_row_rejects_inconsistent_labels():
    validate_row(_valid_row())
    validate_row(_valid_row(class_4way=UNKNOWN, label_binary=1))
    with pytest.raises(ValueError):
        validate_row(_valid_row(class_4way=2, label_binary=0))
    with pytest.raises(ValueError):
        validate_row(_valid_row(class_4way=UNKNOWN, label_binary=UNKNOWN))


# ---------------------------------------------------------------- dedup
BASE = ("The committee reviewed the proposal in detail and concluded that the budget "
        "allocation for the next fiscal year should prioritise infrastructure repair, "
        "public transport, and the expansion of community health clinics in rural areas.")


def test_exact_duplicates():
    hashes = [text_hash(t) for t in [BASE, BASE.upper(), "other text entirely"]]
    assert exact_duplicate_groups(hashes) == {hashes[0]: [0, 1]}


def test_minhash_is_deterministic_and_finds_near_duplicates():
    near = BASE.replace("in detail", "in great detail")
    other = "Photosynthesis converts light energy into chemical energy stored in glucose molecules."
    mh = MinHasher(num_perm=128, seed=7)
    sigs = np.stack([mh.signature(t) for t in (BASE, near, other)])
    assert np.array_equal(sigs, np.stack([MinHasher(128, 7).signature(t) for t in (BASE, near, other)]))
    pairs = near_duplicate_pairs(sigs, bands=32, threshold=0.6)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


# ---------------------------------------------------------------- splitting
def test_components_join_rows_sharing_any_key():
    keys = [["p:1"], ["p:1", "dup:a"], ["dup:a"], ["p:2"]]
    comps = connected_components(keys)
    assert comps[0] == comps[1] == comps[2] != comps[3]


def test_assign_splits_respects_pins_and_groups():
    keys = [["p:1"], ["p:1"], ["p:2"], ["p:2"], ["p:3"]]
    comps = connected_components(keys)
    names = {c: f"c{c}" for c in set(comps)}
    fixed = ["", "validation", "train", "", ""]
    splits, conflicts = assign_splits(comps, names, fixed, {"train": 0.8, "validation": 0.1, "test": 0.1}, 0)
    assert splits[0] == splits[1] == "validation"   # evaluation pin wins for the whole group
    assert splits[2] == splits[3] == "train"
    assert not conflicts
    assert_no_leakage(splits, keys)


def test_assign_splits_reports_conflicting_eval_pins():
    keys = [["p:1"], ["p:1"]]
    comps = connected_components(keys)
    splits, conflicts = assign_splits(comps, {0: "a"}, ["validation", "test"], {"train": 1.0}, 0)
    assert splits == ["", ""] and conflicts


def test_assign_splits_is_reproducible_and_roughly_proportional():
    keys = [[f"p:{i}"] for i in range(5000)]
    comps = connected_components(keys)
    names = {c: f"name{c}" for c in comps}
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    s1, _ = assign_splits(comps, names, [""] * 5000, ratios, seed=42)
    s2, _ = assign_splits(comps, names, [""] * 5000, ratios, seed=42)
    assert s1 == s2
    assert 0.77 < s1.count("train") / 5000 < 0.83


def test_leakage_detected():
    keys = [["p:1"], ["p:1"]]
    assert leakage_report(["train", "test"], keys)["num_leaking_keys"] == 1
    with pytest.raises(LeakageError):
        assert_no_leakage(["train", "test"], keys)


def test_unique_trigram_ratio_flags_loops_only():
    from scriptsense_ml.data.normalization import unique_trigram_ratio
    loop = "it is a film that is not for everyone because " * 20
    assert unique_trigram_ratio(loop) < 0.1
    assert unique_trigram_ratio(BASE + " " + BASE.replace("committee", "council")) > 0.5
    assert unique_trigram_ratio("too short to judge") == 1.0
