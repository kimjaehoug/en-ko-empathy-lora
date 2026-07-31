#!/usr/bin/env python3
"""Train EN Stage1 prototype: LoRA causal LM + affect classification."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.empathy_data import EmpathyCollator, EmpathyJsonlDataset
from src.models.backbone import cuda_mem_gb, git_commit_hash
from src.models.stage1 import Stage1EmpathyModel, build_lora_lm, build_tokenizer
from src.utils.train_log import (
    TrainLogger,
    batch_accuracy,
    estimate_total_steps,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    total_lm = 0.0
    total_em = 0.0
    n = 0
    correct = 0
    counted = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += float(out["loss"])
        total_lm += float(out["lm_loss"])
        if out["emotion_loss"] is not None:
            total_em += float(out["emotion_loss"])
        n += 1
        ids = batch["emotion_ids"]
        valid = ids >= 0
        if valid.any():
            pred = out["emotion_logits"][valid].argmax(dim=-1)
            correct += int((pred == ids[valid]).sum().item())
            counted += int(valid.sum().item())
    model.train()
    return {
        "loss": total_loss / max(n, 1),
        "lm_loss": total_lm / max(n, 1),
        "emotion_loss": total_em / max(n, 1),
        "emotion_acc": (correct / counted) if counted else None,
        "n_batches": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "stage1_en.yaml"),
    )
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    set_seed(int(cfg.get("seed", 42)))
    device = pick_device(str(cfg.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}")

    train_path = ROOT / cfg["train_file"]
    valid_path = ROOT / cfg["valid_file"]
    train_ds = EmpathyJsonlDataset(train_path, max_history=int(cfg.get("max_history", 8)))
    valid_ds = EmpathyJsonlDataset(
        valid_path,
        max_history=int(cfg.get("max_history", 8)),
        emotion_labels=train_ds.emotion_labels,
    )
    print(
        f"train={len(train_ds)} valid={len(valid_ds)} "
        f"n_emotions={len(train_ds.emotion_labels)}"
    )

    tok = build_tokenizer(cfg["model_name"])
    collator = EmpathyCollator(tok, max_length=int(cfg.get("max_length", 384)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        collate_fn=collator,
    )

    lm = build_lora_lm(
        cfg["model_name"],
        lora_r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        dtype=cfg.get("dtype", "bf16"),
        include_mlp=bool(cfg.get("lora_include_mlp", False)),
        target_modules=cfg.get("lora_target_modules"),
    )
    if bool(cfg.get("gradient_checkpointing", False)):
        lm.gradient_checkpointing_enable()
        if hasattr(lm, "enable_input_require_grads"):
            lm.enable_input_require_grads()
    model = Stage1EmpathyModel(
        lm,
        n_emotions=len(train_ds.emotion_labels),
        emotion_loss_weight=float(cfg.get("emotion_loss_weight", 0.2)),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable_params={trainable:,} / total={total:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    out_dir = ROOT / cfg.get("output_dir", "outputs/stage1_en")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "emotion_labels.json").write_text(
        json.dumps(train_ds.emotion_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    max_steps = cfg.get("max_steps")
    eval_every = int(cfg.get("eval_every", 50))
    log_every = int(cfg.get("log_every", 10))
    max_eval_batches = cfg.get("max_eval_batches", 100)
    if max_eval_batches is not None:
        max_eval_batches = int(max_eval_batches)
    grad_accum = int(cfg.get("grad_accum", 1))
    num_epochs = int(cfg.get("num_epochs", 1))
    total_steps = estimate_total_steps(
        n_train=len(train_ds),
        batch_size=int(cfg.get("batch_size", 2)),
        grad_accum=grad_accum,
        num_epochs=num_epochs,
        max_steps=int(max_steps) if max_steps is not None else None,
    )

    logger = TrainLogger(
        stage="Stage1-EN",
        out_dir=out_dir,
        total_steps=total_steps,
        log_every=log_every,
    )
    logger.banner(
        f"model={cfg['model_name']} train={len(train_ds)} valid={len(valid_ds)} "
        f"steps≈{total_steps} eval_every={eval_every}"
    )

    global_step = 0
    optim.zero_grad(set_to_none=True)
    history = []

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"] / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                train_acc = batch_accuracy(
                    out["emotion_logits"], batch.get("emotion_ids"), name="emotion"
                )
                logger.log_train(
                    global_step,
                    loss=float(out["loss"].detach()),
                    lm_loss=float(out["lm_loss"].detach()),
                    extra_losses={
                        "emotion_loss": None
                        if out["emotion_loss"] is None
                        else float(out["emotion_loss"].detach()),
                    },
                    train_acc=train_acc,
                    force=(global_step == 1),
                )
                history.append(
                    {
                        "step": global_step,
                        "loss": float(out["loss"].detach()),
                        "lm_loss": float(out["lm_loss"].detach()),
                        "emotion_loss": None
                        if out["emotion_loss"] is None
                        else float(out["emotion_loss"].detach()),
                        **train_acc,
                    }
                )

                if global_step % eval_every == 0:
                    metrics = evaluate(
                        model, valid_loader, device, max_batches=max_eval_batches
                    )
                    logger.log_eval(global_step, metrics)
                    history.append({"step": global_step, "eval": metrics})

                if max_steps is not None and global_step >= int(max_steps):
                    break
        if max_steps is not None and global_step >= int(max_steps):
            break

    # final eval + save
    metrics = evaluate(model, valid_loader, device, max_batches=max_eval_batches)
    logger.log_final(
        metrics,
        extra={
            "saved_to": str(out_dir),
            "vram_peak_gb": cuda_mem_gb().get("peak_allocated_gb"),
            "steps": global_step,
        },
    )
    history.append({"final_eval": metrics})

    # save peft adapters + emotion head
    model.lm.save_pretrained(out_dir / "lora")
    tok.save_pretrained(out_dir / "tokenizer")
    torch.save(model.emotion_head.state_dict(), out_dir / "emotion_head.pt")
    (out_dir / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "model_name": cfg["model_name"],
        "dtype": cfg.get("dtype", "bf16"),
        "n_emotions": len(train_ds.emotion_labels),
        "trainable_params": trainable,
        "total_params": total,
        "final_eval": metrics,
        "steps": global_step,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "vram": cuda_mem_gb(),
        "git_commit": git_commit_hash(ROOT),
        "seed": int(cfg.get("seed", 42)),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Stage1-EN] progress -> {out_dir / 'progress.json'}", flush=True)


if __name__ == "__main__":
    main()
