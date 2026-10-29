"""Document-level aggregation of calibrated window predictions.

Definitions (also returned to API clients in ``definitions``):
* document class probabilities: mean of calibrated window four-way probabilities, each window
  weighted by the characters it adds beyond the previous window (so overlap is not double counted).
* ai_signal_score: the same weighted mean of the calibrated binary AI-involvement probability.
* flagged_text_share: share of analysed characters lying in windows whose calibrated binary
  probability is >= the deployed threshold. It is a descriptive statistic of model flags, NOT a
  validated estimate of how much of the document was AI-written.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schema import CLASS4_NAMES

UNCERTAIN = "uncertain"
DEFINITIONS = {
    "class_probabilities": "Overlap-weighted mean of temperature-calibrated per-window four-way probabilities.",
    "ai_signal_score": "Overlap-weighted mean of the calibrated binary AI-involvement probability. A model "
                       "signal, not the probability that anyone cheated.",
    "flagged_text_share": "Share of analysed characters inside windows at or above the deployed threshold. "
                          "Descriptive only; not validated as the proportion of AI-written text.",
}


@dataclass
class Policy:
    threshold: float                 # binary AI-involvement threshold chosen on validation human writing
    operating_point: str             # e.g. "fpr_0.01"
    min_eligible_words: int = 150
    min_class_confidence: float = 0.5


def aggregate(window_p4: np.ndarray, window_pai: np.ndarray, weights: np.ndarray,
              eligible_words: int, policy: Policy) -> dict:
    if eligible_words < policy.min_eligible_words or len(weights) == 0 or weights.sum() <= 0:
        return {"status": "insufficient_evidence", "document_class": UNCERTAIN,
                "class_probabilities": None, "ai_signal_score": None, "flagged_text_share": None,
                "reasons": [f"eligible prose ({eligible_words} words) below the {policy.min_eligible_words}-word minimum"]}
    w = weights / weights.sum()
    p4 = (window_p4 * w[:, None]).sum(axis=0)
    pai = float((window_pai * w).sum())
    flagged = window_pai >= policy.threshold
    share = float(w[flagged].sum())

    reasons = []
    if pai >= policy.threshold:
        ai_classes = p4[1:]
        cls = 1 + int(ai_classes.argmax())
        conf = float(ai_classes.max() / max(ai_classes.sum(), 1e-9))
        if conf < policy.min_class_confidence:
            document_class, reasons = UNCERTAIN, ["AI-involvement signal above threshold but the provenance class is ambiguous"]
        else:
            document_class = CLASS4_NAMES[cls].lower()
    else:
        if int(p4.argmax()) == 0:
            document_class = CLASS4_NAMES[0].lower()
        else:
            document_class = UNCERTAIN
            reasons = ["binary signal below threshold but four-way head does not favour human-only"]
    return {"status": "complete", "document_class": document_class,
            "class_probabilities": {n.lower(): float(v) for n, v in zip(CLASS4_NAMES, p4)},
            "ai_signal_score": pai, "flagged_text_share": share, "reasons": reasons}
