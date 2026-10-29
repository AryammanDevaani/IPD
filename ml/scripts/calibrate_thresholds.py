"""Fit temperature scaling and FPR-constrained thresholds on VALIDATION logits only.

Usage: python ml/scripts/calibrate_thresholds.py --run deberta_multitask_s1 [--operating-point fpr_0.01]
Writes <run>/best/calibration.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.evaluation.calibration import apply_temperature, fit_temperature, select_thresholds  # noqa: E402
from scriptsense_ml.evaluation.metrics import MIN_EXPECTED_FALSE_POSITIVES, expected_calibration_error  # noqa: E402
from scriptsense_ml.paths import ARTIFACT_DIR  # noqa: E402

TARGETS = (0.01, 0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--operating-point", default="auto",
                    help="fpr_0.01 | fpr_0.05 | auto (strictest target with enough human validation rows)")
    args = ap.parse_args()
    best = ARTIFACT_DIR / "runs" / args.run / "best"
    z = np.load(best / "val_logits.npz", allow_pickle=True)
    l4, l2, y4, y2 = z["logits4"], z["logits2"], z["y4"], z["y2"]

    t4, t2 = fit_temperature(l4, y4), fit_temperature(l2, y2)
    p_ai = apply_temperature(l2, t2)[:, 1]
    thresholds = select_thresholds(p_ai[y2 != -1], y2[y2 != -1], TARGETS)

    if args.operating_point == "auto":
        ok = [k for k in (f"fpr_{t}" for t in TARGETS)
              if thresholds[k]["expected_false_positives_at_target"] >= MIN_EXPECTED_FALSE_POSITIVES]
        op = ok[0] if ok else None
    else:
        op = args.operating_point
    if op is None:
        raise SystemExit("Not enough human validation rows to support any declared FPR operating point.")

    m4, m2 = y4 != -1, y2 != -1
    ece = {
        "four_way_before": expected_calibration_error(apply_temperature(l4[m4], 1.0), y4[m4])[0],
        "four_way_after": expected_calibration_error(apply_temperature(l4[m4], t4), y4[m4])[0],
        "binary_before": expected_calibration_error(apply_temperature(l2[m2], 1.0), y2[m2])[0],
        "binary_after": expected_calibration_error(apply_temperature(l2[m2], t2), y2[m2])[0],
        "note": "in-sample on the validation set used for fitting; held-out calibration is in evaluate_detector output",
    }
    out = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "fitted_on": "validation split only",
        "temperature_4way": t4, "temperature_binary": t2,
        "thresholds": thresholds,
        "deployed_operating_point": op,
        "deployed_threshold": thresholds[op]["threshold"],
        "validation_ece": ece,
        "n_validation": int(len(y2)),
    }
    (best / "calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
