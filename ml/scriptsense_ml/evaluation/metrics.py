"""Evaluation metrics for the AI-writing detector (CLAUDE.md section 6).

Rows whose label is UNKNOWN for a head are excluded from that head's metrics.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
)

from ..data.schema import CLASS4_NAMES, UNKNOWN

# An FPR operating point is reported as reliable only if it rests on >= this many expected
# false positives (e.g. 1% FPR needs >= 1,000 human samples).
MIN_EXPECTED_FALSE_POSITIVES = 10


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    """Top-label ECE plus the reliability-diagram bins. ``probs`` is (n, k)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            bins.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()),
                         "accuracy": float(correct[m].mean()), "confidence": float(conf[m].mean())})
    return float(ece), bins


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def four_class_metrics(probs4: np.ndarray, y4: np.ndarray) -> dict:
    m = y4 != UNKNOWN
    p, y = probs4[m], y4[m]
    if len(y) == 0:
        return {"n": 0}
    pred = p.argmax(axis=1)
    labels = list(range(len(CLASS4_NAMES)))
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = [i for i in labels if sup[i] > 0]
    ece, bins = expected_calibration_error(p, y)
    return {
        "n": int(len(y)),
        "macro_f1_present_classes": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "classes_present": [CLASS4_NAMES[i] for i in present],
        "accuracy": float((pred == y).mean()),
        "per_class": {CLASS4_NAMES[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                                        "f1": float(f1[i]), "support": int(sup[i])} for i in labels},
        "confusion_matrix": {"labels": CLASS4_NAMES,
                             "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
        "brier": multiclass_brier(p, y),
        "ece": ece,
        "reliability_bins": bins,
    }


def threshold_for_fpr(scores_neg: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t such that mean(scores_neg >= t) <= target_fpr (predict positive if score >= t)."""
    s = np.sort(scores_neg)[::-1]
    k = int(np.floor(target_fpr * len(s)))  # number of negatives allowed above the threshold
    if k >= len(s):
        return float(s[-1])
    return float(np.nextafter(s[k], np.inf))


def binary_metrics(p_ai: np.ndarray, yb: np.ndarray, threshold: float = 0.5,
                   fpr_points: tuple[float, ...] = (0.01, 0.05)) -> dict:
    m = yb != UNKNOWN
    p, y = p_ai[m], yb[m]
    out: dict = {"n": int(len(y)), "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
                 "threshold": threshold}
    if len(y) == 0:
        return out
    pred = (p >= threshold).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    out.update({
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "human_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "ai_recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(np.mean((p - y) ** 2)),
    })
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        ece, bins = expected_calibration_error(np.stack([1 - p, p], axis=1), y)
        out["ece"], out["reliability_bins"] = ece, bins
        # Oracle operating points on THIS set: descriptive only, never used to pick deployment thresholds.
        pts = {}
        for f in fpr_points:
            t = threshold_for_fpr(p[y == 0], f)
            pr = p >= t
            pts[f"fpr_{f}"] = {
                "threshold_on_this_set": t,
                "achieved_fpr": float(pr[y == 0].mean()),
                "recall": float(pr[y == 1].mean()),
                "precision": float((pr & (y == 1)).sum() / max(pr.sum(), 1)),
                "reliable": bool((y == 0).sum() * f >= MIN_EXPECTED_FALSE_POSITIVES),
            }
        out["operating_points_oracle"] = pts
    return out


def recall_by_class_at_threshold(p_ai: np.ndarray, y4: np.ndarray, threshold: float) -> dict:
    """Binary AI-involvement recall broken down by the true four-way class (e.g. paraphrased, mixed)."""
    out = {}
    for i, name in enumerate(CLASS4_NAMES):
        m = y4 == i
        if m.any():
            key = "flag_rate" if i == 0 else "recall"
            out[name] = {key: float((p_ai[m] >= threshold).mean()), "n": int(m.sum())}
    return out


def bootstrap_ci(fn, arrays: tuple[np.ndarray, ...], n_boot: int = 1000, seed: int = 0,
                 groups: np.ndarray | None = None) -> dict:
    """Percentile CI. When ``groups`` is given, resample whole groups (e.g. prompts) not rows."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    if groups is not None:
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        try:
            vals.append(fn(*(a[idx] for a in arrays)))
        except ValueError:  # e.g. a resample with one class for AUROC
            continue
    vals = np.array(vals, dtype=float)
    if len(vals) == 0:
        return {"low": None, "high": None, "n_boot": 0}
    return {"low": float(np.percentile(vals, 2.5)), "high": float(np.percentile(vals, 97.5)),
            "n_boot": int(len(vals))}


def length_bucket(n_words: np.ndarray) -> np.ndarray:
    return np.select([n_words < 50, n_words < 150, n_words < 400], ["<50", "50-149", "150-399"], ">=400")
