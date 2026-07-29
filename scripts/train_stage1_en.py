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
from src.models.stage1 import Stage1EmpathyModel, build_lora_lm, build_tokenizer


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
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    total_lm = 0.0
    total_em = 0.0
    n = 0
    correct = 0
    counted = 0
    for batch in loader:
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
    print(f"device={device}")

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
    )
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
    grad_accum = int(cfg.get("grad_accum", 1))
    num_epochs = int(cfg.get("num_epochs", 1))

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

                if global_step % 5 == 0 or global_step == 1:
                    msg = {
                        "step": global_step,
                        "loss": float(out["loss"].detach()),
                        "lm_loss": float(out["lm_loss"].detach()),
                        "emotion_loss": None
                        if out["emotion_loss"] is None
                        else float(out["emotion_loss"].detach()),
                    }
                    history.append(msg)
                    print(msg)

                if global_step % eval_every == 0:
                    metrics = evaluate(model, valid_loader, device)
                    print({"eval": metrics})
                    history.append({"step": global_step, "eval": metrics})

                if max_steps is not None and global_step >= int(max_steps):
                    break
        if max_steps is not None and global_step >= int(max_steps):
            break

    # final eval + save
    metrics = evaluate(model, valid_loader, device)
    print({"final_eval": metrics})
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
        "n_emotions": len(train_ds.emotion_labels),
        "trainable_params": trainable,
        "total_params": total,
        "final_eval": metrics,
        "steps": global_step,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
