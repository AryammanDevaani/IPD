"""Document-level AI-writing analysis using a trained, calibrated run (CLAUDE.md section 8 contract)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ..data.normalization import normalize_for_model, word_count
from ..evaluation.calibration import apply_temperature
from ..models.multitask import MultiTaskDetector
from .aggregation import DEFINITIONS, Policy, aggregate
from .chunking import eligible_spans, offset_to_page, segment_blocks, token_windows
from .predict import inference_amp_dtype, predict_logits

STANDING_LIMITATIONS = [
    "AI-writing analysis is a probabilistic model signal, not proof of authorship or misconduct.",
    "AI-paraphrased/refined and mixed-authorship text are substantially harder to detect than raw AI output.",
    "Validated only on English text from the training/evaluation domains; other genres and languages may be unreliable.",
    "Results are not comparable to, and are not designed to reproduce, any commercial detector's scores.",
]


class ModelNotReadyError(RuntimeError):
    pass


class AIWritingAnalyzer:
    def __init__(self, run_best_dir: Path, device: str | None = None, min_eligible_words: int = 150):
        run_best_dir = Path(run_best_dir)
        cal_path = run_best_dir / "calibration.json"
        if not (run_best_dir / "heads.pt").exists() or not cal_path.exists():
            raise ModelNotReadyError(f"no trained+calibrated model at {run_best_dir}")
        self.meta = json.loads((run_best_dir / "model_meta.json").read_text())
        self.cal = json.loads(cal_path.read_text())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiTaskDetector.load(run_best_dir).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(run_best_dir / "tokenizer")
        self.max_length = self.meta["config"]["max_length"]
        self.policy = Policy(threshold=self.cal["deployed_threshold"],
                             operating_point=self.cal["deployed_operating_point"],
                             min_eligible_words=min_eligible_words)
        data = self.meta.get("data", {})
        self.model_version = (f"{self.meta['config']['run_name']}"
                              f"+data:{str(data.get('dataset_sha256', ''))[:12]}"
                              f"+cal:{self.cal['fitted_at'][:10]}")
        self.is_subset_model = bool(data.get("is_subset"))

    def analyze(self, text: str, page_starts: list[int] | None = None, document_id: str = "") -> dict:
        page_starts = page_starts or [0]
        blocks = segment_blocks(text)
        spans = eligible_spans(blocks)
        eligible_words = sum(word_count(text[s:e]) for s, e in spans)
        total_words = word_count(text)
        excluded = {}
        for b in blocks:
            if not b.eligible:
                excluded[b.reason] = excluded.get(b.reason, 0) + word_count(text[b.start:b.end])

        windows = [w for span in spans for w in token_windows(text, span, self.tokenizer, self.max_length, 64)]
        limitations = list(STANDING_LIMITATIONS)
        if self.is_subset_model:
            limitations.insert(0, "This model was trained on a documented data SUBSET (smoke/development run); "
                                  "its outputs are not production-validated.")
        base = {
            "document_id": document_id, "model_version": self.model_version,
            "eligible_word_count": eligible_words, "total_word_count": total_words,
            "excluded_word_counts": excluded, "ai_involvement_estimate": None,
            "operating_point": self.policy.operating_point, "threshold": self.policy.threshold,
            "definitions": DEFINITIONS, "limitations": limitations,
        }
        if not windows:
            agg = aggregate(np.zeros((0, 4)), np.zeros(0), np.zeros(0), eligible_words, self.policy)
            return {**base, **agg, "analyzed_word_count": 0, "segments": []}

        chunks = [normalize_for_model(text[w.start:w.end]) for w in windows]
        amp = inference_amp_dtype(self.device)
        l4, l2 = predict_logits(self.model, self.tokenizer, chunks, self.max_length, 16, self.device, amp)
        p4 = apply_temperature(l4, self.cal["temperature_4way"])
        pai = apply_temperature(l2, self.cal["temperature_binary"])[:, 1]
        weights = np.array([w.new_chars for w in windows], dtype=float)
        agg = aggregate(p4, pai, weights, eligible_words, self.policy)

        segments = []
        for w, probs, p in zip(windows, p4, pai):
            flagged = bool(p >= self.policy.threshold)
            segments.append({
                "start": w.start, "end": w.end,
                "page_start": offset_to_page(w.start, page_starts), "page_end": offset_to_page(max(w.end - 1, 0), page_starts),
                "n_tokens": w.n_tokens,
                "class_probabilities": {k.lower(): float(v) for k, v in zip(
                    ["HUMAN_ONLY", "AI_GENERATED", "AI_PARAPHRASED_OR_REFINED", "MIXED_OR_AI_ASSISTED"], probs)},
                "ai_signal": float(p), "flagged": flagged,
                "predicted_class": ["human_only", "ai_generated", "ai_paraphrased_or_refined",
                                    "mixed_or_ai_assisted"][int(probs.argmax())],
                "threshold": self.policy.threshold, "model_version": self.model_version,
            })
        analyzed = word_count(" ".join(text[s:e] for s, e in spans))
        return {**base, **agg, "analyzed_word_count": analyzed, "segments": segments}
