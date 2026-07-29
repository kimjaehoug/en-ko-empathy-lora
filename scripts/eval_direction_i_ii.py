#!/usr/bin/env python3
"""Direction I/II evaluation for Factor-LoRA SELECT baselines.

Direction I  (EN→KO): A/S/R Acc + LM on AI Hub valid
Direction II (return EN): emotion Acc + LM on EmpatheticDialogues valid

EN data used in this repo: EmpatheticDialogues → data/processed/ed_{train,valid,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.empathy_data import EmpathyCollator, EmpathyJsonlDataset
from src.models.encoding import pick_device
from src.models.stage1 import Stage1EmpathyModel, build_tokenizer
from src.models.stage3 import Stage3EmpathyModel


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_stage3(model, loader, device, max_batches: int | None) -> dict:
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
                totals[k] += float(out[k].detach())
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

    metrics = {k: (totals[k] / counts[k] if counts[k] else None) for k in totals}
    for name in correct:
        metrics[f"{name}_acc"] = (
            correct[name] / counted[name] if counted[name] else None
        )
    metrics["n_batches"] = min(len(loader), max_batches or len(loader))
    return metrics


@torch.no_grad()
def eval_stage1(model, loader, device, max_batches: int | None) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ["loss", "lm_loss", "emotion_loss"]}
    counts = {k: 0 for k in totals}
    correct = 0
    counted = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask", "labels", "emotion_ids"}
        }
        out = model(**batch)
        for k in totals:
            if out.get(k) is not None:
                totals[k] += float(out[k].detach())
                counts[k] += 1
        ids = batch["emotion_ids"]
        valid = ids >= 0
        if valid.any():
            pred = out["emotion_logits"][valid].argmax(dim=-1)
            correct += int((pred == ids[valid]).sum().item())
            counted += int(valid.sum().item())

    metrics = {k: (totals[k] / counts[k] if counts[k] else None) for k in totals}
    metrics["emotion_acc"] = correct / counted if counted else None
    metrics["n_batches"] = min(len(loader), max_batches or len(loader))
    return metrics


def load_stage3_bundle(cfg: dict, ckpt_dir: Path, device: torch.device) -> Stage3EmpathyModel:
    labels = json.loads((ckpt_dir / "labels.json").read_text(encoding="utf-8"))
    tok = build_tokenizer(cfg["model_name"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    lm = PeftModel.from_pretrained(base, str(ckpt_dir / "lora"))
    model = Stage3EmpathyModel(
        lm,
        n_emotions=len(labels["emotion"]),
        n_strategies=len(labels["strategy"]),
        n_relations=len(labels["relation"]),
    )
    heads = torch.load(ckpt_dir / "heads.pt", map_location="cpu")
    model.emotion_head.load_state_dict(heads["emotion_head"])
    model.strategy_head.load_state_dict(heads["strategy_head"])
    model.relation_head.load_state_dict(heads["relation_head"])
    model.to(device)
    model.eval()
    return model, tok, labels


def load_stage1_for_en(
    cfg: dict,
    stage1_dir: Path,
    lora_dir: Path,
    device: torch.device,
) -> Stage1EmpathyModel:
    """EN evaluation: given LoRA (stage1 or post-KO) + Stage1 emotion head (32-class ED)."""
    emotion_labels = json.loads((stage1_dir / "emotion_labels.json").read_text())
    tok = build_tokenizer(cfg["model_name"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    lm = PeftModel.from_pretrained(base, str(lora_dir))
    model = Stage1EmpathyModel(lm, n_emotions=len(emotion_labels))
    state = torch.load(stage1_dir / "emotion_head.pt", map_location="cpu")
    model.emotion_head.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tok, emotion_labels


def make_loader(path: Path, tok, cfg: dict, *, lang_hint=None, labels=None, emotion_only=False):
    kwargs = {
        "max_history": int(cfg.get("max_history", 8)),
        "lang_hint": lang_hint,
    }
    if labels is not None:
        kwargs.update(
            emotion_labels=labels.get("emotion"),
            strategy_labels=labels.get("strategy"),
            relation_labels=labels.get("relation"),
        )
    if lang_hint == "ko" or (labels and labels.get("strategy")):
        kwargs["multitask"] = True
    ds = EmpathyJsonlDataset(path, **kwargs)
    collator = EmpathyCollator(tok, max_length=int(cfg.get("max_length", 384)))
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        collate_fn=collator,
    ), ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval_direction_i_ii.yaml"))
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    device = pick_device(str(cfg.get("device", "auto")))
    max_batches = args.max_batches if args.max_batches is not None else cfg.get("max_batches", 50)

    stage1_dir = ROOT / cfg["stage1_dir"]
    report = {
        "en_data": {
            "name": "EmpatheticDialogues (ParlAI)",
            "processed": {
                "train": "data/processed/ed_train.jsonl",
                "valid": "data/processed/ed_valid.jsonl",
                "test": "data/processed/ed_test.jsonl",
            },
            "note": "EN Stage1 / Direction II use these files; A=32 ED emotions.",
        },
        "direction_i_ko": {},
        "direction_ii_en": {},
    }

    # ---- Direction I: KO affective accuracy ----
    print("=== Direction I: KO (AI Hub) ===")
    for name, rel in cfg.get("ko_checkpoints", {}).items():
        ckpt = ROOT / rel
        if not (ckpt / "lora").exists():
            print(f"[skip] {name}: missing {ckpt}")
            report["direction_i_ko"][name] = {"error": f"missing {ckpt}"}
            continue
        model, tok, labels = load_stage3_bundle(cfg, ckpt, device)
        loader, ds = make_loader(
            ROOT / cfg["ko_valid_file"],
            tok,
            cfg,
            lang_hint="ko",
            labels=labels,
        )
        metrics = eval_stage3(model, loader, device, max_batches)
        report["direction_i_ko"][name] = {
            "checkpoint": str(ckpt),
            "n_eval": len(ds),
            "metrics": metrics,
        }
        print(name, {k: metrics[k] for k in ["emotion_acc", "strategy_acc", "relation_acc", "lm_loss"]})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- Direction II: return to EN ----
    print("=== Direction II: EN (EmpatheticDialogues) ===")
    for name, rel in cfg.get("en_lora_variants", {}).items():
        lora_path = ROOT / rel
        if not lora_path.exists():
            print(f"[skip] {name}: missing {lora_path}")
            report["direction_ii_en"][name] = {"error": f"missing {lora_path}"}
            continue
        model, tok, emo_labels = load_stage1_for_en(cfg, stage1_dir, lora_path, device)
        loader, ds = make_loader(
            ROOT / cfg["en_valid_file"],
            tok,
            cfg,
            lang_hint=None,
            labels={"emotion": emo_labels},
        )
        metrics = eval_stage1(model, loader, device, max_batches)
        report["direction_ii_en"][name] = {
            "lora": str(lora_path),
            "emotion_head": str(stage1_dir / "emotion_head.pt"),
            "n_eval": len(ds),
            "n_emotions": len(emo_labels),
            "metrics": metrics,
        }
        print(name, {k: metrics[k] for k in ["emotion_acc", "lm_loss", "loss"]})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # deltas vs EN-before if present
    before = report["direction_ii_en"].get("en_before_ko", {}).get("metrics")
    if before and before.get("emotion_acc") is not None:
        for name, blob in report["direction_ii_en"].items():
            m = blob.get("metrics") or {}
            if m.get("emotion_acc") is None:
                continue
            blob["delta_vs_en_before"] = {
                "emotion_acc": m["emotion_acc"] - before["emotion_acc"],
                "lm_loss": None
                if m.get("lm_loss") is None or before.get("lm_loss") is None
                else m["lm_loss"] - before["lm_loss"],
            }

    out_dir = ROOT / cfg.get("output_dir", "outputs/eval_direction_i_ii")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
