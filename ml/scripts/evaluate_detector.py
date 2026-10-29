"""Evaluate a trained, calibrated run on held-out sets. Never used for selection or thresholds.

Usage: python ml/scripts/evaluate_detector.py --run deberta_multitask_s1 [--sets semeval_a/test txd22/test raid/challenge]
       [--max-rows N]   (documented subsample per set, for smoke runs only)

Every invocation on a final set is appended to <run>/final_eval_audit.jsonl so repeated
peeking at test data is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.data.io import file_sha256  # noqa: E402
from scriptsense_ml.data.schema import CLASS4_NAMES, UNKNOWN  # noqa: E402
from scriptsense_ml.evaluation.calibration import apply_temperature  # noqa: E402
from scriptsense_ml.evaluation.metrics import (  # noqa: E402
    binary_metrics, bootstrap_ci, four_class_metrics, length_bucket, recall_by_class_at_threshold,
)
from scriptsense_ml.inference.predict import inference_amp_dtype, predict_logits  # noqa: E402
from scriptsense_ml.models.multitask import MultiTaskDetector  # noqa: E402
from scriptsense_ml.paths import ARTIFACT_DIR, PROCESSED_DIR  # noqa: E402
from scriptsense_ml.training.data import load_dataset  # noqa: E402

DEFAULT_SETS = ["semeval_a/test", "txd22/test", "raid/challenge"]
SLICES = ["domain", "generator", "attack_or_edit_type", "length_bucket"]


def _macro_f1(y4, p4):
    m = y4 != UNKNOWN
    return f1_score(y4[m], p4[m].argmax(1), average="macro")


def _fpr(y2, pai, t):
    neg = y2 == 0
    if not neg.any():
        raise ValueError("no negatives")
    return float((pai[neg] >= t).mean())


def evaluate_set(df, p4, pai, cal) -> dict:
    t = cal["deployed_threshold"]
    y4, y2 = df.class_4way.to_numpy(), df.label_binary.to_numpy()
    groups = (df.prompt_id.where(df.prompt_id != "", df.original_document_id)
              .where(lambda s: s != "", df.row_id)).to_numpy()
    res = {"n": len(df), "four_way": four_class_metrics(p4, y4),
           "binary": binary_metrics(pai, y2, threshold=t),
           "binary_flag_rate_by_true_class": recall_by_class_at_threshold(pai, y4, t),
           "deployed_operating_point": cal["deployed_operating_point"], "threshold": t,
           "ci95_grouped_bootstrap": {}}
    m4 = y4 != UNKNOWN
    if m4.sum() and len(np.unique(y4[m4])) > 1:
        res["ci95_grouped_bootstrap"]["macro_f1_4way"] = bootstrap_ci(_macro_f1, (y4, p4), 500, 0, groups)
    m2 = y2 != UNKNOWN
    if len(np.unique(y2[m2])) == 2:
        res["ci95_grouped_bootstrap"]["auroc"] = bootstrap_ci(
            roc_auc_score, (y2[m2], pai[m2]), 500, 0, groups[m2])
        res["ci95_grouped_bootstrap"]["human_fpr"] = bootstrap_ci(
            lambda a, b: _fpr(a, b, t), (y2[m2], pai[m2]), 500, 0, groups[m2])
    df = df.assign(length_bucket=length_bucket(df.text.str.split().str.len().to_numpy()))
    res["slices"] = {}
    for col in SLICES:
        res["slices"][col] = {}
        for val, idx in df.groupby(col).indices.items():
            if len(idx) < 30:
                continue
            b = binary_metrics(pai[idx], y2[idx], threshold=t)
            f = four_class_metrics(p4[idx], y4[idx])
            res["slices"][col][str(val) or "unknown"] = {
                "n": int(len(idx)), "human_fpr": b.get("human_false_positive_rate"), "ai_recall": b.get("ai_recall"),
                "auroc": b.get("auroc"), "macro_f1_4way": f.get("macro_f1_present_classes"), "accuracy_4way": f.get("accuracy")}
    return res


def plot(res: dict, name: str, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm = np.array(res["four_way"]["confusion_matrix"]["matrix"])
    axes[0].imshow(cm, cmap="Blues")
    short = ["HUMAN", "AI_GEN", "AI_PARA", "MIXED"]
    axes[0].set_xticks(range(4), short, rotation=30)
    axes[0].set_yticks(range(4), short)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    axes[0].set_title(f"{name}: 4-way confusion")
    bins = res["binary"].get("reliability_bins", [])
    if bins:
        axes[1].plot([0, 1], [0, 1], "--", color="grey")
        axes[1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-")
        axes[1].set_xlabel("confidence")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("binary reliability")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name.replace('/', '_')}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--dataset", default=str(PROCESSED_DIR / "dataset.parquet"))
    args = ap.parse_args()

    run = ARTIFACT_DIR / "runs" / args.run
    best = run / "best"
    cal = json.loads((best / "calibration.json").read_text())
    meta = json.loads((best / "model_meta.json").read_text())
    max_len = meta["config"]["max_length"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = inference_amp_dtype(device)
    model = MultiTaskDetector.load(best).to(device)
    tok = AutoTokenizer.from_pretrained(best / "tokenizer")

    df = load_dataset(Path(args.dataset))
    out_dir = run / "eval"
    out_dir.mkdir(exist_ok=True)
    summary = []
    for name in args.sets:
        src, split = name.split("/")
        if split == "train":
            raise SystemExit("refusing to report training-split metrics as held-out results")
        part = df[(df.source_dataset == src) & (df.split == split)]
        subsampled = args.max_rows is not None and len(part) > args.max_rows
        if subsampled:
            part = part.sample(n=args.max_rows, random_state=0)
        if part.empty:
            print(f"skip {name}: no rows")
            continue
        l4, l2 = predict_logits(model, tok, part.text.tolist(), max_len, 32, device, amp)
        p4 = apply_temperature(l4, cal["temperature_4way"])
        pai = apply_temperature(l2, cal["temperature_binary"])[:, 1]
        res = evaluate_set(part.reset_index(drop=True), p4, pai, cal)
        res.update({"set": name, "subsampled_to": args.max_rows if subsampled else None,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                    "dataset_sha256": file_sha256(Path(args.dataset)), "passage_level": True,
                    "note": f"passage-level: each row truncated to {max_len} tokens"})
        if src == "raid" and split == "challenge":
            unseen = part.generator.isin(["cohere", "cohere-chat"]).to_numpy()
            res["unseen_generator_subset"] = {
                "binary": binary_metrics(pai[unseen], part.label_binary.to_numpy()[unseen], cal["deployed_threshold"])}
            res["unseen_generator_subset"]["binary"].pop("reliability_bins", None)
        (out_dir / f"{name.replace('/', '_')}.json").write_text(json.dumps(res, indent=2, default=str))
        plot(res, name, out_dir)
        b, f = res["binary"], res["four_way"]
        row = {"set": name, "n": res["n"], "subsampled": subsampled,
               "macro_f1_4way": f.get("macro_f1_present_classes"), "auroc": b.get("auroc"),
               "auprc": b.get("auprc"), "human_fpr": b.get("human_false_positive_rate"),
               "ai_recall": b.get("ai_recall"), "brier_binary": b.get("brier"), "ece_binary": b.get("ece"),
               "ece_4way": f.get("ece")}
        for cname in CLASS4_NAMES[1:]:
            row[f"recall_{cname}"] = f.get("per_class", {}).get(cname, {}).get("recall")
        summary.append(row)
        with open(run / "final_eval_audit.jsonl", "a") as fh:
            fh.write(json.dumps({"set": name, "at": res["evaluated_at"], "n": res["n"]}) + "\n")
        print(json.dumps(row, default=str))
    pd.DataFrame(summary).to_csv(out_dir / "metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
