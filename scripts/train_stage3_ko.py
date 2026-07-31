#!/usr/bin/env python3
"""Train KO Stage3: gated LoRA adaptation + A/S/R multitask heads.

v2 technical upgrades (paper-oriented):
  - Strategy multi-label BCE (AI Hub S is a set, not a single class)
  - Soft-share EN LoRA init + LoRA-anchor regularization (Dir II retention)
  - EN EmpatheticDialogues replay batches interleaved with KO
  - Inverse-frequency pos_weight / class weights
"""

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
from src.models.encoding import pick_device
from src.models.stage1 import build_tokenizer
from src.models.stage3 import (
    Stage3EmpathyModel,
    build_stage3_lm,
    lora_anchor_loss,
    snapshot_lora_params,
)
from src.utils.train_log import (
    TrainLogger,
    batch_accuracy,
    estimate_total_steps,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def inverse_freq_weights(ids: list[int], n_classes: int) -> torch.Tensor:
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for i in ids:
        if 0 <= i < n_classes:
            counts[i] += 1
    counts = counts.clamp(min=1.0)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


def multilabel_pos_weight(multihots: list[torch.Tensor], n_classes: int) -> torch.Tensor:
    pos = torch.zeros(n_classes, dtype=torch.float32)
    neg = torch.zeros(n_classes, dtype=torch.float32)
    for m in multihots:
        pos += m.sum(dim=0)
        neg += (1.0 - m).sum(dim=0)
    return (neg / pos.clamp(min=1.0)).clamp(max=50.0)


def gate_scaled_loss_weights(cfg: dict, gates: dict) -> dict[str, float]:
    """SELECT-only: share → downweight KO CE; relearn → upweight."""
    emo = float(cfg.get("emotion_loss_weight", 0.2))
    strat = float(cfg.get("strategy_loss_weight", 1.0))
    rel = float(cfg.get("relation_loss_weight", 0.2))
    if not bool(cfg.get("gate_conditioned_losses", False)):
        return {"emotion": emo, "strategy": strat, "relation": rel}

    share_scale = float(cfg.get("gate_share_loss_scale", 0.55))
    relearn_scale = float(cfg.get("gate_relearn_loss_scale", 1.75))
    suppress_scale = float(cfg.get("gate_suppress_loss_scale", 0.8))
    scales = {"share": share_scale, "relearn": relearn_scale, "suppress": suppress_scale}

    emo *= scales.get((gates.get("affect") or {}).get("decision", "relearn"), 1.0)
    strat *= scales.get((gates.get("strategy") or {}).get("decision", "relearn"), 1.0)
    rel *= scales.get((gates.get("relation") or {}).get("decision", "relearn"), 1.0)
    return {"emotion": emo, "strategy": strat, "relation": rel}


def apply_curriculum_weights(
    base: dict[str, float],
    gates: dict,
    *,
    step: int,
    total_steps: int,
    enabled: bool,
) -> dict[str, float]:
    """Early phase: push relearn factors harder; late phase: restore base."""
    if not enabled or total_steps <= 0:
        return base
    frac = step / max(total_steps, 1)
    out = dict(base)
    if frac < 0.4:
        # emphasize relearn heads
        if (gates.get("strategy") or {}).get("decision") == "relearn":
            out["strategy"] *= 1.35
        if (gates.get("relation") or {}).get("decision") == "relearn":
            out["relation"] *= 1.2
        if (gates.get("affect") or {}).get("decision") == "share":
            out["emotion"] *= 0.75
    return out


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ["loss", "lm_loss", "emotion_loss", "strategy_loss", "relation_loss"]}
    counts = {k: 0 for k in totals}
    correct = {k: 0 for k in ["emotion", "relation"]}
    counted = {k: 0 for k in correct}

    # multi-label strategy stats
    s_tp = s_fp = s_fn = s_correct_bits = s_total_bits = 0
    s_exact = s_n = 0

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
            ("relation", "relation_logits", "relation_ids"),
        ]:
            ids = batch[ids_key]
            valid = ids >= 0
            if valid.any():
                pred = out[logits_key][valid].argmax(dim=-1)
                correct[name] += int((pred == ids[valid]).sum().item())
                counted[name] += int(valid.sum().item())

        # Strategy: prefer multi-label metrics
        if "strategy_multihot" in batch:
            target = batch["strategy_multihot"]
            pred = (out["strategy_logits"].sigmoid() >= 0.5).float()
            row_has = target.sum(dim=-1) > 0
            if row_has.any():
                t = target[row_has]
                p = pred[row_has]
                s_tp += int(((p == 1) & (t == 1)).sum().item())
                s_fp += int(((p == 1) & (t == 0)).sum().item())
                s_fn += int(((p == 0) & (t == 1)).sum().item())
                s_correct_bits += int((p == t).sum().item())
                s_total_bits += int(t.numel())
                s_exact += int((p == t).all(dim=-1).sum().item())
                s_n += int(row_has.sum().item())
        else:
            ids = batch["strategy_ids"]
            valid = ids >= 0
            if valid.any():
                pred = out["strategy_logits"][valid].argmax(dim=-1)
                # stash into emotion-style counters via micro approx
                s_exact += int((pred == ids[valid]).sum().item())
                s_n += int(valid.sum().item())

    model.train()
    metrics = {k: (totals[k] / counts[k] if counts[k] else None) for k in totals}
    for name in correct:
        metrics[f"{name}_acc"] = (
            correct[name] / counted[name] if counted[name] else None
        )

    if s_total_bits > 0:
        prec = s_tp / max(s_tp + s_fp, 1)
        rec = s_tp / max(s_tp + s_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics["strategy_acc"] = f1  # report micro-F1 as primary Acc for Dir I S
        metrics["strategy_micro_f1"] = f1
        metrics["strategy_precision"] = prec
        metrics["strategy_recall"] = rec
        metrics["strategy_hamming_acc"] = s_correct_bits / s_total_bits
        metrics["strategy_exact_match"] = s_exact / max(s_n, 1)
    elif s_n > 0:
        metrics["strategy_acc"] = s_exact / s_n
    else:
        metrics["strategy_acc"] = None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage3_ko.yaml"))
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--init_mode",
        choices=["auto", "share", "relearn", "suppress", "affect_priority", "soft_share", "select"],
        default=None,
        help="Override LoRA init (select=EN share, no suppress; soft_share=EN+optional suppress)",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gates_file", type=str, default=None)
    parser.add_argument(
        "--lm_only",
        action="store_true",
        help="Zero A/S/R head loss weights (Stage3 LM-only ablation A3).",
    )
    parser.add_argument("--en_replay_every", type=int, default=None)
    parser.add_argument("--lora_anchor_weight", type=float, default=None)
    parser.add_argument(
        "--gate_conditioned_losses",
        action="store_true",
        help="Scale A/S/R loss weights by Stage2 gate decisions (SELECT-only).",
    )
    parser.add_argument(
        "--no_gate_conditioned_losses",
        action="store_true",
        help="Disable gate-conditioned loss scaling even if config enables it.",
    )
    parser.add_argument(
        "--select_curriculum",
        action="store_true",
        help="Early steps emphasize relearn factors (S/R); later restore balance.",
    )
    parser.add_argument(
        "--no_select_curriculum",
        action="store_true",
        help="Disable curriculum even if config enables it (Blind/Scratch).",
    )
    parser.add_argument(
        "--freeze_lora",
        action="store_true",
        help="Freeze EN LoRA; train A/S/R heads only (factor-modular SELECT).",
    )
    parser.add_argument(
        "--strategy_scope",
        choices=["utterance", "session"],
        default=None,
        help="Strategy label source: utterance (last listener) or session axes.",
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
    if args.en_replay_every is not None:
        cfg["en_replay_every"] = args.en_replay_every
    if args.lora_anchor_weight is not None:
        cfg["lora_anchor_weight"] = args.lora_anchor_weight
    if args.gate_conditioned_losses:
        cfg["gate_conditioned_losses"] = True
    if args.no_gate_conditioned_losses:
        cfg["gate_conditioned_losses"] = False
    if args.select_curriculum:
        cfg["select_curriculum"] = True
    if args.no_select_curriculum:
        cfg["select_curriculum"] = False
    if args.freeze_lora:
        cfg["freeze_lora"] = True
    if args.strategy_scope is not None:
        cfg["strategy_scope"] = args.strategy_scope
    if args.lm_only:
        cfg["emotion_loss_weight"] = 0.0
        cfg["strategy_loss_weight"] = 0.0
        cfg["relation_loss_weight"] = 0.0

    set_seed(int(cfg.get("seed", 42)))
    device = pick_device(str(cfg.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)

    gates_path = ROOT / cfg["gates_file"]
    gates_blob = json.loads(gates_path.read_text(encoding="utf-8"))
    gates = gates_blob["gates"]

    strategy_scope = str(cfg.get("strategy_scope", "utterance"))
    train_ds = EmpathyJsonlDataset(
        ROOT / cfg["train_file"],
        max_history=int(cfg.get("max_history", 8)),
        multitask=True,
        lang_hint="ko",
        strategy_scope=strategy_scope,
    )
    valid_ds = EmpathyJsonlDataset(
        ROOT / cfg["valid_file"],
        max_history=int(cfg.get("max_history", 8)),
        multitask=True,
        lang_hint="ko",
        emotion_labels=train_ds.emotion_labels,
        strategy_labels=train_ds.strategy_labels,
        relation_labels=train_ds.relation_labels,
        strategy_scope=strategy_scope,
    )
    print(
        f"train={len(train_ds)} valid={len(valid_ds)} "
        f"A={len(train_ds.emotion_labels)} S={len(train_ds.strategy_labels)} "
        f"R={len(train_ds.relation_labels)} strategy_scope={strategy_scope}",
        flush=True,
    )

    tok = build_tokenizer(cfg["model_name"])
    collator = EmpathyCollator(
        tok,
        max_length=int(cfg.get("max_length", 384)),
        n_strategies=len(train_ds.strategy_labels),
    )
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

    # Optional EN replay for Direction II retention
    en_replay_every = int(cfg.get("en_replay_every", 0))
    en_loader_iter = None
    if en_replay_every > 0 and cfg.get("en_replay_file"):
        en_ds = EmpathyJsonlDataset(
            ROOT / cfg["en_replay_file"],
            max_history=int(cfg.get("max_history", 8)),
            lang_hint=None,
        )
        en_collator = EmpathyCollator(tok, max_length=int(cfg.get("max_length", 384)))
        en_loader = DataLoader(
            en_ds,
            batch_size=int(cfg.get("batch_size", 2)),
            shuffle=True,
            collate_fn=en_collator,
        )
        en_loader_iter = iter(en_loader)
        print(f"EN replay: every {en_replay_every} KO steps, n={len(en_ds)}", flush=True)

    lm, init_mode = build_stage3_lm(
        model_name=cfg["model_name"],
        stage1_lora_dir=str(ROOT / cfg["stage1_lora_dir"]),
        gates=gates,
        lora_r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        force_init=cfg.get("force_init"),
        dtype=cfg.get("dtype", "bf16"),
        include_mlp=bool(cfg.get("lora_include_mlp", False)),
        target_modules=cfg.get("lora_target_modules"),
    )
    if bool(cfg.get("gradient_checkpointing", False)):
        lm.gradient_checkpointing_enable()
        if hasattr(lm, "enable_input_require_grads"):
            lm.enable_input_require_grads()
    print(f"lora_init_mode={init_mode}", flush=True)
    print(f"gates={ {k: v['decision'] for k, v in gates.items()} }", flush=True)

    freeze_lora = bool(cfg.get("freeze_lora", False))
    if freeze_lora:
        n_frozen = 0
        for n, p in lm.named_parameters():
            if "lora_" in n:
                p.requires_grad = False
                n_frozen += 1
        print(f"freeze_lora=True frozen_lora_tensors={n_frozen}", flush=True)
        init_mode = f"{init_mode}_heads_only"

    # Class / pos weights from train distribution
    emo_w = inverse_freq_weights(
        [r["emotion_id"] for r in train_ds.rows], len(train_ds.emotion_labels)
    )
    rel_w = inverse_freq_weights(
        [r["relation_id"] for r in train_ds.rows], len(train_ds.relation_labels)
    )
    # Build multihot for pos_weight
    n_s = len(train_ds.strategy_labels)
    mh_list = []
    for r in train_ds.rows:
        m = torch.zeros(n_s)
        for sid in r.get("strategy_ids_multi") or (
            [r["strategy_id"]] if r["strategy_id"] >= 0 else []
        ):
            if 0 <= sid < n_s:
                m[sid] = 1.0
        mh_list.append(m)
    s_pos_w = multilabel_pos_weight(mh_list, n_s)
    print(f"strategy_pos_weight={s_pos_w.tolist()}", flush=True)

    loss_w = gate_scaled_loss_weights(cfg, gates)
    print(
        f"loss_weights emotion={loss_w['emotion']:.3f} "
        f"strategy={loss_w['strategy']:.3f} relation={loss_w['relation']:.3f} "
        f"gate_conditioned={bool(cfg.get('gate_conditioned_losses', False))} "
        f"curriculum={bool(cfg.get('select_curriculum', False))}",
        flush=True,
    )

    model = Stage3EmpathyModel(
        lm,
        n_emotions=len(train_ds.emotion_labels),
        n_strategies=len(train_ds.strategy_labels),
        n_relations=len(train_ds.relation_labels),
        emotion_loss_weight=loss_w["emotion"],
        strategy_loss_weight=loss_w["strategy"],
        relation_loss_weight=loss_w["relation"],
        strategy_multilabel=bool(cfg.get("strategy_multilabel", True)),
        deep_strategy_head=bool(cfg.get("deep_strategy_head", True)),
        emotion_class_weights=emo_w if cfg.get("use_class_weights", True) else None,
        strategy_pos_weight=s_pos_w if cfg.get("use_class_weights", True) else None,
        relation_class_weights=rel_w if cfg.get("use_class_weights", True) else None,
    ).to(device)

    # SELECT with frozen EN LoRA: no LoRA anchor (nothing trainable in LoRA)
    anchor_w = float(cfg.get("lora_anchor_weight", 0.0))
    lora_snap = None
    if anchor_w > 0 and "share" in init_mode and not freeze_lora:
        lora_snap = snapshot_lora_params(model.lm)
        print(f"LoRA anchor enabled weight={anchor_w} n_params={len(lora_snap)}", flush=True)

    share_n = sum(v["decision"] == "share" for v in gates.values())
    base_lr = float(cfg.get("lr", 2e-4))
    head_lr = float(cfg.get("head_lr", base_lr))
    if freeze_lora:
        # Heads-only SELECT: use full head_lr (do not apply share_lr_scale)
        pass
    elif share_n >= 2 or "soft_share" in init_mode or "share" in init_mode or "select" in init_mode:
        base_lr *= float(cfg.get("share_lr_scale", 0.5))
        head_lr = float(cfg.get("head_lr", base_lr))

    # Param groups: optionally higher LR on relearn heads (S/R)
    head_params = {
        "emotion": list(model.emotion_head.parameters()),
        "strategy": list(model.strategy_head.parameters()),
        "relation": list(model.relation_head.parameters()),
    }
    lora_params = [p for n, p in model.lm.named_parameters() if p.requires_grad]
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": base_lr})
    # Gate-aware head LRs: share → lower, relearn → higher
    head_lr_scale = {
        "share": float(cfg.get("head_share_lr_scale", 0.5)),
        "relearn": float(cfg.get("head_relearn_lr_scale", 1.5)),
        "suppress": float(cfg.get("head_suppress_lr_scale", 0.75)),
    }
    gate_to_head = {
        "affect": "emotion",
        "strategy": "strategy",
        "relation": "relation",
    }
    for gate_name, head_name in gate_to_head.items():
        dec = (gates.get(gate_name) or {}).get("decision", "relearn")
        scale = head_lr_scale.get(dec, 1.0) if bool(cfg.get("gate_conditioned_losses", False)) else 1.0
        param_groups.append(
            {"params": head_params[head_name], "lr": head_lr * scale}
        )

    optim = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"lr_lora={base_lr} lr_head={head_lr} freeze_lora={freeze_lora} "
        f"trainable={trainable:,}",
        flush=True,
    )

    out_dir = ROOT / cfg.get("output_dir", "outputs/stage3_ko")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(
        json.dumps(
            {
                "emotion": train_ds.emotion_labels,
                "strategy": train_ds.strategy_labels,
                "relation": train_ds.relation_labels,
                "strategy_multilabel": bool(cfg.get("strategy_multilabel", True)),
                "strategy_scope": strategy_scope,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    max_steps = cfg.get("max_steps")
    eval_every = int(cfg.get("eval_every", 25))
    log_every = int(cfg.get("log_every", 10))
    max_eval_batches = cfg.get("max_eval_batches", 50)
    grad_accum = int(cfg.get("grad_accum", 4))
    num_epochs = int(cfg.get("num_epochs", 1))
    total_steps = estimate_total_steps(
        n_train=len(train_ds),
        batch_size=int(cfg.get("batch_size", 2)),
        grad_accum=grad_accum,
        num_epochs=num_epochs,
        max_steps=int(max_steps) if max_steps is not None else None,
    )

    stage_label = f"Stage3-KO({init_mode})"
    logger = TrainLogger(
        stage=stage_label,
        out_dir=out_dir,
        total_steps=total_steps,
        log_every=log_every,
    )
    logger.banner(
        f"model={cfg['model_name']} train={len(train_ds)} valid={len(valid_ds)} "
        f"init={init_mode} steps≈{total_steps} S=multilabel "
        f"gates={ {k: v['decision'] for k, v in gates.items()} }"
    )

    global_step = 0
    history = []
    optim.zero_grad(set_to_none=True)
    model.train()

    if max_steps is not None and int(max_steps) <= 0:
        print({"note": "max_steps<=0; saving initialization without KO updates"}, flush=True)
    else:
        for _epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                # Curriculum / gate weights (SELECT): update head loss scales online
                cur_w = apply_curriculum_weights(
                    loss_w,
                    gates,
                    step=global_step,
                    total_steps=total_steps,
                    enabled=bool(cfg.get("select_curriculum", False)),
                )
                model.emotion_loss_weight = cur_w["emotion"]
                model.strategy_loss_weight = cur_w["strategy"]
                model.relation_loss_weight = cur_w["relation"]

                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out["loss"]
                if lora_snap is not None and anchor_w > 0:
                    a_loss = lora_anchor_loss(model.lm, lora_snap)
                    loss = loss + anchor_w * a_loss
                    out_anchor = float(a_loss.detach())
                else:
                    out_anchor = None
                (loss / grad_accum).backward()

                if (batch_idx + 1) % grad_accum == 0:
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                    # EN replay (Dir II): LM-only on ED.
                    # Do NOT pass ED emotion_ids (32-class) into KO affect head (6-class).
                    if (
                        en_loader_iter is not None
                        and en_replay_every > 0
                        and global_step % en_replay_every == 0
                    ):
                        try:
                            en_batch = next(en_loader_iter)
                        except StopIteration:
                            en_loader_iter = iter(en_loader)
                            en_batch = next(en_loader_iter)
                        en_batch = {
                            k: v.to(device)
                            for k, v in en_batch.items()
                            if k in {"input_ids", "attention_mask", "labels"}
                        }
                        # Explicitly skip A/S/R heads on EN replay
                        bsz = en_batch["input_ids"].size(0)
                        en_batch["emotion_ids"] = torch.full(
                            (bsz,), -1, dtype=torch.long, device=device
                        )
                        en_batch["strategy_ids"] = torch.full(
                            (bsz,), -1, dtype=torch.long, device=device
                        )
                        en_batch["relation_ids"] = torch.full(
                            (bsz,), -1, dtype=torch.long, device=device
                        )
                        en_out = model(**en_batch)
                        en_out["loss"].backward()
                        optim.step()
                        optim.zero_grad(set_to_none=True)

                    train_acc = {}
                    train_acc.update(
                        batch_accuracy(
                            out["emotion_logits"],
                            batch.get("emotion_ids"),
                            name="emotion",
                        )
                    )
                    if "strategy_multihot" in batch:
                        with torch.no_grad():
                            pred = (out["strategy_logits"].sigmoid() >= 0.5).float()
                            tgt = batch["strategy_multihot"]
                            row = tgt.sum(dim=-1) > 0
                            if row.any():
                                train_acc["train_strategy_f1"] = float(
                                    (
                                        ((pred[row] == 1) & (tgt[row] == 1)).sum()
                                        / (
                                            ((pred[row] == 1) | (tgt[row] == 1))
                                            .sum()
                                            .clamp(min=1)
                                        )
                                    ).item()
                                )
                    else:
                        train_acc.update(
                            batch_accuracy(
                                out["strategy_logits"],
                                batch.get("strategy_ids"),
                                name="strategy",
                            )
                        )
                    train_acc.update(
                        batch_accuracy(
                            out["relation_logits"],
                            batch.get("relation_ids"),
                            name="relation",
                        )
                    )
                    extra = {
                        "strategy_loss": None
                        if out["strategy_loss"] is None
                        else float(out["strategy_loss"].detach()),
                        "relation_loss": None
                        if out["relation_loss"] is None
                        else float(out["relation_loss"].detach()),
                    }
                    if out_anchor is not None:
                        extra["anchor"] = out_anchor
                    logger.log_train(
                        global_step,
                        loss=float(out["loss"].detach()),
                        lm_loss=float(out["lm_loss"].detach()),
                        extra_losses=extra,
                        train_acc=train_acc,
                        force=(global_step == 1),
                    )
                    history.append({"step": global_step, "loss": float(out["loss"].detach()), **train_acc})

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

    metrics = evaluate(model, valid_loader, device, max_batches=max_eval_batches)
    logger.log_final(
        metrics,
        extra={
            "saved_to": str(out_dir),
            "init_mode": init_mode,
            "vram_peak_gb": cuda_mem_gb().get("peak_allocated_gb"),
            "steps": global_step,
        },
    )
    history.append({"final_eval": metrics})

    model.lm.save_pretrained(out_dir / "lora")
    tok.save_pretrained(out_dir / "tokenizer")
    torch.save(
        {
            "emotion_head": model.emotion_head.state_dict(),
            "strategy_head": model.strategy_head.state_dict(),
            "relation_head": model.relation_head.state_dict(),
            "strategy_multilabel": bool(cfg.get("strategy_multilabel", True)),
        },
        out_dir / "heads.pt",
    )
    (out_dir / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "model_name": cfg["model_name"],
                "dtype": cfg.get("dtype", "bf16"),
                "init_mode": init_mode,
                "gates": gates,
                "lr": base_lr,
                "trainable_params": trainable,
                "final_eval": metrics,
                "steps": global_step,
                "upgrades": {
                    "strategy_multilabel": bool(cfg.get("strategy_multilabel", True)),
                    "strategy_scope": strategy_scope,
                    "en_replay_every": en_replay_every,
                    "lora_anchor_weight": anchor_w,
                    "deep_strategy_head": bool(cfg.get("deep_strategy_head", True)),
                    "gate_conditioned_losses": bool(
                        cfg.get("gate_conditioned_losses", False)
                    ),
                    "select_curriculum": bool(cfg.get("select_curriculum", False)),
                    "freeze_lora": freeze_lora,
                    "loss_weights": loss_w,
                },
                "device": str(device),
                "cuda_device_name": torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None,
                "vram": cuda_mem_gb(),
                "git_commit": git_commit_hash(ROOT),
                "seed": int(cfg.get("seed", 42)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{stage_label}] progress -> {out_dir / 'progress.json'}", flush=True)


if __name__ == "__main__":
    main()
