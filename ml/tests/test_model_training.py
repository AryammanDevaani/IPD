import json

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import DebertaV2Config

from scriptsense_ml.data.schema import UNKNOWN
from scriptsense_ml.evaluation.calibration import apply_temperature, fit_temperature, select_thresholds
from scriptsense_ml.evaluation.metrics import (
    binary_metrics, bootstrap_ci, expected_calibration_error, four_class_metrics, threshold_for_fpr,
)
from scriptsense_ml.models.multitask import MultiTaskDetector, class_weights_from_counts, multitask_loss
from scriptsense_ml.training.data import apply_ablation, stratified_subset

TOKENIZER = "microsoft/deberta-v3-base"


def tiny_config():
    return DebertaV2Config(vocab_size=128100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=512, relative_attention=True,
                           position_buckets=256, pos_att_type=["p2c", "c2p"], norm_rel_ebd="layer_norm",
                           max_relative_positions=-1, share_att_key=True, type_vocab_size=0)


# ------------------------------------------------------------------ loss
def test_unknown_labels_do_not_train_the_four_way_head():
    l4 = torch.randn(3, 4, requires_grad=True)
    l2 = torch.randn(3, 2, requires_grad=True)
    y4 = torch.tensor([UNKNOWN, UNKNOWN, UNKNOWN])
    y2 = torch.tensor([1, 0, 1])
    loss, parts = multitask_loss(l4, l2, y4, y2)
    loss.backward()
    assert parts["loss_4way"] == 0.0
    assert torch.all(l4.grad == 0)
    assert l2.grad.abs().sum() > 0


def test_binary_only_mode_ignores_four_way_head():
    l4 = torch.randn(2, 4, requires_grad=True)
    loss, _ = multitask_loss(l4, torch.randn(2, 2), torch.tensor([0, 1]), torch.tensor([0, 1]), four_way_weight=0.0)
    loss.backward()
    assert torch.all(l4.grad == 0)


def test_class_weights():
    w = class_weights_from_counts({0: 100, 1: 10000, 3: 400}, power=0.5)
    assert w[2] == 0 and w[0] > w[3] > w[1]
    assert torch.isclose(w[w > 0].mean(), torch.tensor(1.0))


# ------------------------------------------------------------------ calibration / metrics
def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 4000)
    true_logits = rng.normal(size=(4000, 4))
    true_logits[np.arange(4000), y] += 1.5
    # sample labels from the true distribution, then present over-confident logits (x3)
    p = apply_temperature(true_logits, 1.0)
    y = np.array([rng.choice(4, p=pi) for pi in p])
    t = fit_temperature(true_logits * 3.0, y)
    assert 2.5 < t < 3.5


def test_threshold_for_fpr_controls_false_positive_rate():
    neg = np.linspace(0, 1, 1000)
    for f in (0.01, 0.05, 0.2):
        t = threshold_for_fpr(neg, f)
        assert (neg >= t).mean() <= f + 1e-9
    thr = select_thresholds(np.r_[neg, np.ones(10)], np.r_[np.zeros(1000), np.ones(10)])
    assert thr["fpr_0.01"]["validation_fpr"] <= 0.01 and thr["fpr_0.01"]["n_human_validation"] == 1000


def test_metrics_exclude_unknown_and_report_classes():
    p4 = np.eye(4)[[0, 1, 2, 3, 1]]
    y4 = np.array([0, 1, 2, UNKNOWN, 2])
    m = four_class_metrics(p4, y4)
    assert m["n"] == 4 and m["per_class"]["AI_PARAPHRASED_OR_REFINED"]["recall"] == 0.5
    b = binary_metrics(np.array([0.1, 0.9, 0.8, 0.2]), np.array([0, 1, 1, UNKNOWN]))
    assert b["n"] == 3 and b["human_false_positive_rate"] == 0.0 and b["auroc"] == 1.0
    ece, _ = expected_calibration_error(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([1, 0]))
    assert ece == 0.0


def test_grouped_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 600), 0, 1)
    from sklearn.metrics import roc_auc_score
    ci = bootstrap_ci(roc_auc_score, (y, p), n_boot=200, groups=np.arange(600) // 3)
    assert ci["low"] <= roc_auc_score(y, p) <= ci["high"]


# ------------------------------------------------------------------ data selection
def _df(n=200):
    return pd.DataFrame({"row_id": [f"r{i}" for i in range(n)], "text": ["t"] * n,
                         "source_dataset": ["raid"] * (n // 2) + ["txd22"] * (n // 2),
                         "attack_or_edit_type": ["paraphrase", "none"] * (n // 2),
                         "class_4way": [2, 1, 3, 0] * (n // 4), "label_binary": [1, 1, 1, 0] * (n // 4)})


def test_ablation_filters():
    df = _df()
    out, removed = apply_ablation(df, {"exclude_raid_paraphrase": True, "exclude_txd22_mixed": True})
    assert removed["raid_paraphrase"] == 50
    assert not ((out.source_dataset == "raid") & (out.attack_or_edit_type == "paraphrase")).any()
    assert not ((out.source_dataset == "txd22") & out.class_4way.isin([2, 3])).any()


def test_stratified_subset_is_deterministic_and_proportional():
    df = _df(400)
    a, b = stratified_subset(df, 100, 3), stratified_subset(df, 100, 3)
    assert a.row_id.tolist() == b.row_id.tolist() and 90 <= len(a) <= 110
    assert stratified_subset(df, None, 3) is df


# ------------------------------------------------------------------ training smoke (CPU, tiny random encoder)
def test_training_smoke_tiny_encoder(tmp_path, monkeypatch):
    from scriptsense_ml.training import trainer

    real_init = MultiTaskDetector.__init__

    def tiny_init(self, encoder_name, dropout=0.1, encoder_config=None):
        real_init(self, encoder_name, dropout, encoder_config=tiny_config())

    monkeypatch.setattr(MultiTaskDetector, "__init__", tiny_init)
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "light", "paper"]

    def make(n, prefix):
        return pd.DataFrame({
            "row_id": [f"{prefix}{i}" for i in range(n)],
            "text": [" ".join(rng.choice(words, 20)) for _ in range(n)],
            "class_4way": [[0, 1, 2, 3, UNKNOWN][i % 5] for i in range(n)],
            "label_binary": [[0, 1, 1, 1, 1][i % 5] for i in range(n)],
            "source_dataset": "synthetic_fixture"})

    cfg = dict(run_name="smoke", model_name="tiny", tokenizer_name=TOKENIZER, mode="multitask", seed=1,
               max_length=32, learning_rate=1e-3, epochs=2, per_device_batch_size=4, grad_accum_steps=2,
               weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0, binary_loss_weight=0.5,
               class_weight_power=0.5, precision="fp32", gradient_checkpointing=False,
               early_stopping_patience=5, eval_every_steps=None, selection_metric="auto", eval_batch_size=8)
    meta = trainer.train(cfg, make(40, "t"), make(10, "v"), tmp_path / "run", {"fixture": "synthetic"})
    best = tmp_path / "run" / "best"
    assert meta["best"] is not None and (best / "heads.pt").exists()
    assert json.loads((best / "model_meta.json").read_text())["label_map_4way"]["2"] == "AI_PARAPHRASED_OR_REFINED"
    logits = np.load(best / "val_logits.npz")
    assert logits["logits4"].shape == (10, 4) and logits["logits2"].shape == (10, 2)
    reloaded = MultiTaskDetector.load(best)
    assert reloaded.head4.out_features == 4
    log = [json.loads(line) for line in open(tmp_path / "run" / "train_log.jsonl")]
    assert any(r["event"] == "eval" for r in log)


def _smoke_cfg(**kw):
    cfg = dict(run_name="smoke", model_name="tiny", tokenizer_name=TOKENIZER, mode="multitask", seed=1,
               max_length=32, learning_rate=1e-3, epochs=2, per_device_batch_size=4, grad_accum_steps=2,
               weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0, binary_loss_weight=0.5,
               class_weight_power=0.5, precision="fp32", gradient_checkpointing=False,
               early_stopping_patience=5, eval_every_steps=None, selection_metric="auto", eval_batch_size=8)
    cfg.update(kw)
    return cfg


def _frames():
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "river", "stone", "light", "paper"]

    def make(n, prefix):
        return pd.DataFrame({
            "row_id": [f"{prefix}{i}" for i in range(n)],
            "text": [" ".join(rng.choice(words, 20)) for _ in range(n)],
            "class_4way": [[0, 1, 2, 3, UNKNOWN][i % 5] for i in range(n)],
            "label_binary": [[0, 1, 1, 1, 1][i % 5] for i in range(n)],
            "source_dataset": "synthetic_fixture"})
    return make(40, "t"), make(10, "v")


def test_interrupted_and_resumed_training_matches_uninterrupted(tmp_path, monkeypatch):
    """Checkpoint/resume must be exact: same final weights as a run that never stopped."""
    from scriptsense_ml.training import trainer

    real_init = MultiTaskDetector.__init__
    monkeypatch.setattr(MultiTaskDetector, "__init__",
                        lambda self, n, dropout=0.1, encoder_config=None: real_init(self, n, dropout, tiny_config()))
    tr, va = _frames()
    prov = {"train_row_ids_sha256": "abc"}

    trainer.train(_smoke_cfg(), tr, va, tmp_path / "full", prov)
    # interrupt mid-epoch-1 (step 7 of 10, epoch 0 has 5 steps), then resume with the same command
    out = trainer.train(_smoke_cfg(checkpoint_every_steps=3, stop_after_steps=7), tr, va, tmp_path / "res", prov)
    assert out["interrupted_at_step"] == 7 and (tmp_path / "res" / "last_checkpoint.pt").exists()
    meta = trainer.train(_smoke_cfg(checkpoint_every_steps=3), tr, va, tmp_path / "res", prov)
    assert meta["optimizer_steps_done"] == 10 and len(meta["resumed_at"]) == 1
    assert not (tmp_path / "res" / "last_checkpoint.pt").exists()  # cleaned up after finishing

    a = torch.load(tmp_path / "full" / "best" / "heads.pt", weights_only=True)
    b = torch.load(tmp_path / "res" / "best" / "heads.pt", weights_only=True)
    for k in ("head4", "head2"):
        for name in a[k]:
            assert torch.equal(a[k][name], b[k][name]), f"{k}.{name} differs after resume"

    with pytest.raises(SystemExit, match="already finished"):
        trainer.train(_smoke_cfg(), tr, va, tmp_path / "res", prov)


def test_resume_refuses_different_data_or_config(tmp_path, monkeypatch):
    from scriptsense_ml.training import trainer

    real_init = MultiTaskDetector.__init__
    monkeypatch.setattr(MultiTaskDetector, "__init__",
                        lambda self, n, dropout=0.1, encoder_config=None: real_init(self, n, dropout, tiny_config()))
    tr, va = _frames()
    trainer.train(_smoke_cfg(stop_after_steps=2), tr, va, tmp_path / "r", {"train_row_ids_sha256": "abc"})
    with pytest.raises(SystemExit, match="different rows"):
        trainer.train(_smoke_cfg(), tr, va, tmp_path / "r", {"train_row_ids_sha256": "zzz"})
    with pytest.raises(SystemExit, match="learning_rate"):
        trainer.train(_smoke_cfg(learning_rate=5e-4), tr, va, tmp_path / "r", {"train_row_ids_sha256": "abc"})


def test_precision_never_picks_emulated_bf16(monkeypatch):
    from scriptsense_ml.training import trainer
    monkeypatch.setattr(trainer, "native_bf16_supported", lambda: False)  # e.g. Colab T4
    assert trainer.resolve_precision("auto", "cuda") == (torch.float16, True)
    monkeypatch.setattr(trainer, "native_bf16_supported", lambda: True)
    assert trainer.resolve_precision("auto", "cuda") == (torch.bfloat16, False)
    assert trainer.resolve_precision("auto", "cpu") == (None, False)
