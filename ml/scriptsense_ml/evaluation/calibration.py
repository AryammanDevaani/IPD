"""Temperature scaling (Guo et al., 2017) fit on validation logits only, and
FPR-constrained threshold selection on held-out human validation writing."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from ..data.schema import UNKNOWN
from .metrics import threshold_for_fpr


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    m = y != UNKNOWN
    z, t = logits[m].astype(np.float64), y[m]
    if len(t) == 0:
        return 1.0

    def nll(log_t: float) -> float:
        return -log_softmax(z / np.exp(log_t), axis=1)[np.arange(len(t)), t].mean()

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits.astype(np.float64) / temperature, axis=1)


def select_thresholds(p_ai_val: np.ndarray, yb_val: np.ndarray, target_fprs=(0.01, 0.05)) -> dict:
    """Thresholds on the calibrated binary AI probability that keep the *validation* human
    false-positive rate at or below each target. Reports the support so callers can see when
    an operating point rests on too few human samples."""
    neg = p_ai_val[yb_val == 0]
    pos = p_ai_val[yb_val == 1]
    out = {}
    for f in target_fprs:
        t = threshold_for_fpr(neg, f)
        out[f"fpr_{f}"] = {
            "threshold": t,
            "validation_fpr": float((neg >= t).mean()) if len(neg) else None,
            "validation_recall": float((pos >= t).mean()) if len(pos) else None,
            "n_human_validation": int(len(neg)),
            "expected_false_positives_at_target": float(len(neg) * f),
        }
    return out
