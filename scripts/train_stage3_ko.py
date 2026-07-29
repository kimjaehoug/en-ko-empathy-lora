#!/usr/bin/env python3
"""Train KO Stage3: gated LoRA adaptation + A/S/R multitask heads."""

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
from src.models.encoding import pick_device
from src.models.stage1 import build_tokenizer
from src.models.stage3 import Stage3EmpathyModel, build_stage3_lm


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ["loss", "lm_loss", "emotion_loss", "strategy_loss", "relation_loss"]}
    counts = {k: 0 for k in totals}
    correct = {k: 0 for k in ["emotion", "strategy", "relation"]}
    counted = {k: 0 for k in correct}

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        for k in totals:
            if out.get(k) is not None:
                totals[k] += float(out[k].detach() if torch.is_tensor(out[k]) else out[k])
                counts[k] += 1
        for name, logits_key, ids_key in [
            ("emotion", "emotion_logits", "emotion_ids"),
            ("strategy", "strategy_logits", "strategy_ids"),
            ("relation", "relation_logits", "relation_ids"),
        ]:
            ids = batch[ids_key]
            valid = ids >= 0
            if valid.any():
                pred = out[logits_key][valid].argmax(dim=-1)
                correct[name] += int((pred == ids[valid]).sum().item())
                counted[name] += int(valid.sum().item())

    model.train()
    metrics = {
        k: (totals[k] / counts[k] if counts[k] else None) for k in totals
    }
    for name in correct:
        metrics[f"{name}_acc"] = (
            correct[name] / counted[name] if counted[name] else None
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage3_ko.yaml"))
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--init_mode",
        choices=["auto", "share", "relearn", "suppress", "affect_priority"],
        default=None,
        help="Override LoRA init: share/relearn/suppress/affect_priority/auto",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gates_file", type=str, default=None)
    parser.add_argument(
        "--lm_only",
        action="store_true",
        help="Zero A/S/R head loss weights (Stage3 LM-only ablation A3).",
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.init_mode is not None:
        cfg["force_init"] = args.init_mode
    if args.gates_file is not None:
        cfg["gates_file"] = args.gates_file
    if args.lm_only:
        cfg["emotion_loss_weight"] = 0.0
        cfg["strategy_loss_weight"] = 0.0
        cfg["relation_loss_weight"] = 0.0

    set_seed(int(cfg.get("seed", 42)))
    device = pick_device(str(cfg.get("device", "auto")))

    gates_path = ROOT / cfg["gates_file"]
    gates_blob = json.loads(gates_path.read_text(encoding="utf-8"))
    gates = gates_blob["gates"]

    train_ds = EmpathyJsonlDataset(
        ROOT / cfg["train_file"],
        max_history=int(cfg.get("max_history", 8)),
        multitask=True,
        lang_hint="ko",
    )
    valid_ds = EmpathyJsonlDataset(
        ROOT / cfg["valid_file"],
        max_history=int(cfg.get("max_history", 8)),
        multitask=True,
        lang_hint="ko",
        emotion_labels=train_ds.emotion_labels,
        strategy_labels=train_ds.strategy_labels,
        relation_labels=train_ds.relation_labels,
    )
    print(
        f"train={len(train_ds)} valid={len(valid_ds)} "
        f"A={len(train_ds.emotion_labels)} S={len(train_ds.strategy_labels)} "
        f"R={len(train_ds.relation_labels)}"
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

    lm, init_mode = build_stage3_lm(
        model_name=cfg["model_name"],
        stage1_lora_dir=str(ROOT / cfg["stage1_lora_dir"]),
        gates=gates,
        lora_r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        force_init=cfg.get("force_init"),
    )
    print(f"lora_init_mode={init_mode}")
    print(f"gates={ {k: v['decision'] for k, v in gates.items()} }")

    model = Stage3EmpathyModel(
        lm,
        n_emotions=len(train_ds.emotion_labels),
        n_strategies=len(train_ds.strategy_labels),
        n_relations=len(train_ds.relation_labels),
        emotion_loss_weight=float(cfg.get("emotion_loss_weight", 0.2)),
        strategy_loss_weight=float(cfg.get("strategy_loss_weight", 0.3)),
        relation_loss_weight=float(cfg.get("relation_loss_weight", 0.2)),
    ).to(device)

    # LR schedule by gate: share -> smaller LR
    share_n = sum(v["decision"] == "share" for v in gates.values())
    base_lr = float(cfg.get("lr", 2e-4))
    if share_n >= 2:
        base_lr *= float(cfg.get("share_lr_scale", 0.5))

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"lr={base_lr} trainable={trainable:,}")

    out_dir = ROOT / cfg.get("output_dir", "outputs/stage3_ko")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(
        json.dumps(
            {
                "emotion": train_ds.emotion_labels,
                "strategy": train_ds.strategy_labels,
                "relation": train_ds.relation_labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    max_steps = cfg.get("max_steps")
    eval_every = int(cfg.get("eval_every", 25))
    max_eval_batches = cfg.get("max_eval_batches", 50)
    grad_accum = int(cfg.get("grad_accum", 4))
    num_epochs = int(cfg.get("num_epochs", 1))

    global_step = 0
    history = []
    optim.zero_grad(set_to_none=True)
    model.train()

    # max_steps == 0: save initialization only (A4 Stage1-only / no KO train)
    if max_steps is not None and int(max_steps) <= 0:
        print({"note": "max_steps<=0; saving initialization without KO updates"})
    else:
        for _epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                (out["loss"] / grad_accum).backward()
                if (batch_idx + 1) % grad_accum == 0:
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % 5 == 0 or global_step == 1:
                        msg = {
                            "step": global_step,
                            "loss": float(out["loss"].detach()),
                            "lm_loss": float(out["lm_loss"].detach()),
                            "strategy_loss": None
                            if out["strategy_loss"] is None
                            else float(out["strategy_loss"].detach()),
                            "relation_loss": None
                            if out["relation_loss"] is None
                            else float(out["relation_loss"].detach()),
                        }
                        history.append(msg)
                        print(msg)
                    if global_step % eval_every == 0:
                        metrics = evaluate(
                            model, valid_loader, device, max_batches=max_eval_batches
                        )
                        print({"eval": metrics})
                        history.append({"step": global_step, "eval": metrics})
                    if max_steps is not None and global_step >= int(max_steps):
                        break
            if max_steps is not None and global_step >= int(max_steps):
                break

    metrics = evaluate(model, valid_loader, device, max_batches=max_eval_batches)
    print({"final_eval": metrics})
    history.append({"final_eval": metrics})

    model.lm.save_pretrained(out_dir / "lora")
    tok.save_pretrained(out_dir / "tokenizer")
    torch.save(
        {
            "emotion_head": model.emotion_head.state_dict(),
            "strategy_head": model.strategy_head.state_dict(),
            "relation_head": model.relation_head.state_dict(),
        },
        out_dir / "heads.pt",
    )
    (out_dir / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "init_mode": init_mode,
                "gates": gates,
                "lr": base_lr,
                "trainable_params": trainable,
                "final_eval": metrics,
                "steps": global_step,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
