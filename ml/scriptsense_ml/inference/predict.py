"""Batched logit prediction shared by training-time validation, evaluation and serving."""
from __future__ import annotations

import numpy as np
import torch


def native_bf16_supported() -> bool:
    """True only for GPUs with hardware bf16 (Ampere+). torch.cuda.is_bf16_supported() also
    returns True when bf16 is merely emulated (e.g. Colab T4), which is very slow."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def inference_amp_dtype(device: str):
    if not device.startswith("cuda"):
        return None
    return torch.bfloat16 if native_bf16_supported() else torch.float16


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32,
                   device: str = "cpu", amp_dtype=None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    out4, out2 = [], []
    # Sort by length so batches pad little; restore order at the end.
    order = np.argsort([len(t) for t in texts])
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
            l4, l2 = model(enc["input_ids"], enc["attention_mask"])
        out4.append(l4.float().cpu().numpy())
        out2.append(l2.float().cpu().numpy())
    l4 = np.concatenate(out4) if out4 else np.zeros((0, 4))
    l2 = np.concatenate(out2) if out2 else np.zeros((0, 2))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return l4[inv], l2[inv]
