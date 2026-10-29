"""Encoder with a four-way provenance head and an auxiliary binary AI-involvement head.

Works with any Hugging Face encoder (microsoft/deberta-v3-base primary, roberta-base baseline).
Labels equal to -1 (UNKNOWN) are ignored by the corresponding loss, so a row that only
supports the binary label never pushes the four-way head toward a guessed class.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from ..data.schema import BINARY_NAMES, CLASS4_NAMES, UNKNOWN


class MultiTaskDetector(nn.Module):
    def __init__(self, encoder_name: str, dropout: float = 0.1, encoder_config=None):
        super().__init__()
        if encoder_config is not None:  # tests: tiny randomly initialised encoder
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            # fp32 master weights: transformers>=5 otherwise keeps the checkpoint dtype, and the
            # deberta-v3-base checkpoint is stored in fp16, which diverges to NaN when trained.
            self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.encoder_name = encoder_name
        self.dropout = nn.Dropout(dropout)
        self.head4 = nn.Linear(hidden, len(CLASS4_NAMES))
        self.head2 = nn.Linear(hidden, len(BINARY_NAMES))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.head4(cls), self.head2(cls)

    def save(self, path: Path, metadata: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path / "encoder")
        torch.save({"head4": self.head4.state_dict(), "head2": self.head2.state_dict()}, path / "heads.pt")
        meta = {**metadata, "encoder_name": self.encoder_name,
                "label_map_4way": dict(enumerate(CLASS4_NAMES)), "label_map_binary": dict(enumerate(BINARY_NAMES))}
        (path / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, path: Path, map_location="cpu") -> "MultiTaskDetector":
        meta = json.loads((path / "model_meta.json").read_text())
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.encoder = AutoModel.from_pretrained(path / "encoder", dtype=torch.float32)
        model.encoder_name = meta["encoder_name"]
        hidden = model.encoder.config.hidden_size
        model.dropout = nn.Dropout(0.0)
        model.head4 = nn.Linear(hidden, len(CLASS4_NAMES))
        model.head2 = nn.Linear(hidden, len(BINARY_NAMES))
        heads = torch.load(path / "heads.pt", map_location=map_location, weights_only=True)
        model.head4.load_state_dict(heads["head4"])
        model.head2.load_state_dict(heads["head2"])
        return model


def multitask_loss(logits4, logits2, y4, y2, class_weights4=None, binary_weight: float = 0.5,
                   four_way_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Weighted sum of masked cross-entropies. ``four_way_weight=0`` gives a binary-only baseline."""
    ce4 = nn.CrossEntropyLoss(weight=class_weights4, ignore_index=UNKNOWN, reduction="sum")
    ce2 = nn.CrossEntropyLoss(ignore_index=UNKNOWN, reduction="sum")
    n4, n2 = (y4 != UNKNOWN).sum(), (y2 != UNKNOWN).sum()
    # Normalise by the (weighted) count of supervised rows, guarding all-unknown batches.
    if class_weights4 is not None:
        denom4 = class_weights4[y4[y4 != UNKNOWN]].sum().clamp(min=1e-8)
    else:
        denom4 = n4.clamp(min=1)
    l4 = ce4(logits4, y4) / denom4 if n4 > 0 else logits4.sum() * 0.0
    l2 = ce2(logits2, y2) / n2.clamp(min=1) if n2 > 0 else logits2.sum() * 0.0
    total = four_way_weight * l4 + binary_weight * l2
    return total, {"loss_4way": float(l4.detach()), "loss_binary": float(l2.detach())}


def class_weights_from_counts(counts: dict[int, int], power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights tempered by ``power`` (0.5 = inverse sqrt), normalised to mean 1
    over present classes. Absent classes get weight 0."""
    w = torch.zeros(len(CLASS4_NAMES))
    for c, n in counts.items():
        if c != UNKNOWN and n > 0:
            w[c] = float(n) ** -power
    present = w > 0
    w[present] = w[present] / w[present].mean()
    return w
