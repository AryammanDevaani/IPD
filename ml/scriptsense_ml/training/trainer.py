"""Training loop for the multi-task detector.

Selection uses validation only. Final test/challenge data are never loaded here.
"""
from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..data.schema import UNKNOWN
from ..evaluation.metrics import binary_metrics, four_class_metrics
from ..inference.predict import native_bf16_supported, predict_logits
from ..models.multitask import MultiTaskDetector, class_weights_from_counts, multitask_loss
from .data import TextDataset

PACKAGES = ["torch", "transformers", "tokenizers", "sentencepiece", "numpy", "pandas", "scikit-learn", "scipy"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_precision(precision: str, device: str):
    if not device.startswith("cuda") or precision == "fp32":
        return None, False
    # torch.cuda.is_bf16_supported() also counts *emulated* bf16 (e.g. a Colab T4), which is very
    # slow, so "auto" only picks bf16 on GPUs with native support (compute capability >= 8).
    if precision == "bf16" or (precision == "auto" and native_bf16_supported()):
        return torch.bfloat16, False
    return torch.float16, True  # fp16 needs a GradScaler


def environment_info() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            versions[p] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = None
    return {"python": platform.python_version(), "packages": versions, "git_commit": commit,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def evaluate(model, tokenizer, df, cfg, device, amp_dtype) -> tuple[dict, np.ndarray, np.ndarray]:
    l4, l2 = predict_logits(model, tokenizer, df.text.tolist(), cfg["max_length"], cfg["eval_batch_size"],
                            device, amp_dtype)
    p4 = torch.softmax(torch.from_numpy(l4), 1).numpy()
    p2 = torch.softmax(torch.from_numpy(l2), 1).numpy()[:, 1]
    y4, y2 = df.class_4way.to_numpy(), df.label_binary.to_numpy()
    res = {"overall": {"four_way": four_class_metrics(p4, y4), "binary": binary_metrics(p2, y2)}}
    for src, idx in df.groupby("source_dataset").indices.items():
        res[src] = {"four_way": four_class_metrics(p4[idx], y4[idx]), "binary": binary_metrics(p2[idx], y2[idx])}
    for blk in res.values():  # keep the training log compact
        blk["four_way"].pop("reliability_bins", None)
        blk["binary"].pop("reliability_bins", None)
    return res, l4, l2


def selection_score(res: dict, cfg: dict) -> float:
    metric = cfg["selection_metric"]
    if metric == "auto":
        metric = "macro_f1_4way" if cfg["mode"] == "multitask" else "auroc"
    if metric == "macro_f1_4way":
        return res["overall"]["four_way"].get("macro_f1_present_classes", float("nan"))
    return res["overall"]["binary"].get(metric, float("nan"))


# Config keys that must match for a checkpoint to be resumed.
RESUME_KEYS = ["model_name", "mode", "seed", "max_length", "learning_rate", "epochs", "per_device_batch_size",
               "grad_accum_steps", "weight_decay", "warmup_ratio", "binary_loss_weight", "class_weight_power",
               "freeze_word_embeddings", "ablation", "max_train_rows", "max_val_rows"]
MAX_NONFINITE_STEPS = 20


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a disconnect mid-write never corrupts the previous checkpoint


def load_tokenizer_cached(name: str, cache_root: Path):
    """DeBERTa-v3 converts its sentencepiece model to a fast tokenizer on every load (~1 min);
    save the converted tokenizer once and reuse it."""
    cache = cache_root / name.replace("/", "__")
    if (cache / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(cache)
    tok = AutoTokenizer.from_pretrained(name)
    tok.save_pretrained(cache)
    return tok


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """Deterministic shuffle per epoch, so a resumed run sees exactly the same batches."""
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()


def train(cfg: dict, train_df, val_df, run_dir: Path, data_provenance: dict, encoder_config=None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last_checkpoint.pt"
    meta_path = run_dir / "run_meta.json"
    resume = cfg.get("resume", True) and ckpt_path.exists()
    if not resume and meta_path.exists() and json.loads(meta_path.read_text()).get("finished_at"):
        raise SystemExit(f"run '{cfg['run_name']}' already finished in {run_dir}; use a new run_name")

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype, use_scaler = resolve_precision(cfg["precision"], device)

    tokenizer = load_tokenizer_cached(cfg.get("tokenizer_name") or cfg["model_name"], run_dir.parent / "_tokenizer_cache")
    model = MultiTaskDetector(cfg["model_name"], encoder_config=encoder_config).to(device)
    if cfg.get("gradient_checkpointing") and device == "cuda":
        model.encoder.gradient_checkpointing_enable()
    if cfg.get("freeze_word_embeddings"):
        model.encoder.get_input_embeddings().requires_grad_(False)

    four_way_weight = 1.0 if cfg["mode"] == "multitask" else 0.0
    counts = Counter(int(c) for c in train_df.class_4way if c != UNKNOWN)
    cw = class_weights_from_counts(counts, cfg["class_weight_power"]).to(device) if four_way_weight else None

    ds = TextDataset(train_df)
    bs, accum = cfg["per_device_batch_size"], cfg["grad_accum_steps"]
    batches_per_epoch = math.ceil(len(ds) / bs)
    workers = 0 if os.name == "nt" else int(cfg.get("dataloader_workers", 0))

    def collate(batch):
        texts, y4, y2 = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=cfg["max_length"], padding=True, return_tensors="pt")
        return enc, torch.tensor(y4), torch.tensor(y2)

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    steps_per_epoch = math.ceil(batches_per_epoch / accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    state = {"epoch": 0, "batch": 0, "step": 0, "best": None, "best_score": -float("inf"),
             "bad_evals": 0, "stop": False, "elapsed_s": 0.0}
    if resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["train_row_ids_sha256"] != data_provenance.get("train_row_ids_sha256"):
            raise SystemExit("checkpoint was trained on different rows; rebuild the same dataset or use a new run_name")
        changed = [k for k in RESUME_KEYS if ck["config"].get(k) != cfg.get(k)]
        if changed:
            raise SystemExit(f"checkpoint config differs on {changed}; use the original settings or a new run_name")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        _set_rng_state(ck["rng"])
        meta = json.loads(meta_path.read_text())
        meta.setdefault("resumed_at", []).append({"at": datetime.now(timezone.utc).isoformat(), "step": state["step"]})
        print(f"[resume] epoch={state['epoch']} batch={state['batch']}/{batches_per_epoch} "
              f"step={state['step']}/{total_steps}", flush=True)
    else:
        meta = {
            "config": cfg, "environment": environment_info(), "device": device,
            "precision": str(amp_dtype) if amp_dtype else "fp32",
            "started_at": datetime.now(timezone.utc).isoformat(), "data": data_provenance,
            "train_class_counts_4way": {int(k): v for k, v in sorted(counts.items())},
            "train_binary_counts": {int(k): int(v) for k, v in train_df.label_binary.value_counts().items()},
            "effective_class_weights_4way": cw.tolist() if cw is not None else None,
            "optimizer_steps_planned": total_steps,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
        }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    log = open(run_dir / "train_log.jsonl", "a", encoding="utf-8")
    t_start = time.time()

    def elapsed() -> float:
        return round(state["elapsed_s"] + time.time() - t_start, 1)

    def save_checkpoint() -> None:
        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                      "scaler": scaler.state_dict(), "state": {**state, "elapsed_s": elapsed()}, "rng": _rng_state(),
                      "config": cfg, "train_row_ids_sha256": data_provenance.get("train_row_ids_sha256")}, ckpt_path)

    def do_eval(epoch):
        res, l4, l2 = evaluate(model, tokenizer, val_df, cfg, device, amp_dtype)
        score = selection_score(res, cfg)
        rec = {"event": "eval", "epoch": epoch, "step": state["step"], "selection_score": score,
               "elapsed_s": elapsed(), "validation": res}
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()
        print(f"[eval] epoch={epoch} step={state['step']} score={score:.4f}", flush=True)
        if score > state["best_score"]:
            state["best_score"], state["bad_evals"] = score, 0
            state["best"] = {"epoch": epoch, "step": state["step"], "score": score}
            model.save(run_dir / "best", {**{k: meta[k] for k in ("config", "environment", "data", "precision")},
                                          "selected_at": state["best"],
                                          "tokenizer_name": cfg.get("tokenizer_name") or cfg["model_name"]})
            tokenizer.save_pretrained(run_dir / "best" / "tokenizer")
            np.savez_compressed(run_dir / "best" / "val_logits.npz", logits4=l4, logits2=l2,
                                y4=val_df.class_4way.to_numpy(), y2=val_df.label_binary.to_numpy(),
                                row_id=val_df.row_id.to_numpy())
        else:
            state["bad_evals"] += 1
            if state["bad_evals"] > cfg["early_stopping_patience"]:
                state["stop"] = True
        model.train()

    eval_every = cfg.get("eval_every_steps")
    ckpt_every = int(cfg.get("checkpoint_every_steps") or 0)
    stop_after = cfg.get("stop_after_steps")  # simulates an interruption (tests / debugging)
    nonfinite = 0

    for epoch in range(state["epoch"], cfg["epochs"]):
        if state["stop"]:
            break
        start = state["batch"] if epoch == state["epoch"] else 0
        order = epoch_order(len(ds), cfg["seed"], epoch)[start * bs:]
        loader = DataLoader(Subset(ds, order), batch_size=bs, shuffle=False, collate_fn=collate, num_workers=workers)
        model.train()
        opt.zero_grad(set_to_none=True)
        for j, (enc, y4, y2) in enumerate(loader):
            i = start + j  # batch index within the epoch
            enc, y4, y2 = enc.to(device), y4.to(device), y2.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                l4, l2 = model(enc["input_ids"], enc["attention_mask"])
            loss, parts = multitask_loss(l4.float(), l2.float(), y4, y2, cw, cfg["binary_loss_weight"], four_way_weight)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum and i + 1 != batches_per_epoch:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state.update(step=state["step"] + 1, epoch=epoch, batch=i + 1)
            step = state["step"]

            nonfinite = 0 if torch.isfinite(loss) else nonfinite + 1
            if nonfinite >= MAX_NONFINITE_STEPS:
                log.close()
                raise RuntimeError(f"loss non-finite for {nonfinite} consecutive steps at step {step}; training diverged")
            if step % 50 == 0 or step == 1:
                rec = {"event": "train", "epoch": epoch, "step": step, "of_steps": total_steps,
                       "loss": float(loss.detach()), **parts, "lr": sched.get_last_lr()[0], "elapsed_s": elapsed()}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[train] {rec}", flush=True)
            if eval_every and step % eval_every == 0:
                do_eval(epoch)
            if ckpt_every and step % ckpt_every == 0:
                save_checkpoint()
            if stop_after and step >= stop_after:
                save_checkpoint()
                log.close()
                return {**meta, "interrupted_at_step": step}
            if state["stop"]:
                break
        if state["stop"]:
            break
        if not eval_every:
            do_eval(epoch)
        state.update(epoch=epoch + 1, batch=0)
        if ckpt_every:
            save_checkpoint()

    log.close()
    meta.update({"finished_at": datetime.now(timezone.utc).isoformat(), "best": state["best"],
                 "optimizer_steps_done": state["step"], "early_stopped": state["stop"], "train_seconds": elapsed()})
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    if ckpt_path.exists() and not cfg.get("keep_last_checkpoint"):
        ckpt_path.unlink()  # ~2 GB; the selected model lives in best/
    return meta
