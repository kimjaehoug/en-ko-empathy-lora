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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.empathy_data import EmpathyCollator, EmpathyJsonlDataset
from src.models.backbone import load_base_causal_lm
from src.models.encoding import pick_device
from src.models.stage1 import Stage1EmpathyModel, build_tokenizer
from src.models.stage3 import Stage3EmpathyModel, set_active_adapters


def resolve_stage3_lora_dir(ckpt_dir: Path) -> Path:
    """Return LoRA dir for Stage3 checkpoints (MAD-X stores task adapter under lora/task/)."""
    task = ckpt_dir / "lora" / "task"
    if (task / "adapter_model.safetensors").exists() or (task / "adapter_config.json").exists():
        return task
    return ckpt_dir / "lora"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_stage3(model, loader, device, max_batches: int | None) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ["loss", "lm_loss", "emotion_loss", "strategy_loss", "relation_loss"]}
    counts = {k: 0 for k in totals}
    correct = {k: 0 for k in ["emotion", "relation"]}
    counted = {k: 0 for k in correct}
    s_tp = s_fp = s_fn = s_correct_bits = s_total_bits = 0
    s_exact = s_n = 0
    s_single_correct = s_single_n = 0
    n_examples = 0
    n_batches_run = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n_batches_run += 1
        n_examples += int(batch["input_ids"].size(0))
        for k in totals:
            if out.get(k) is not None:
                totals[k] += float(out[k].detach())
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
                s_single_correct += int((pred == ids[valid]).sum().item())
                s_single_n += int(valid.sum().item())

    metrics = {k: (totals[k] / counts[k] if counts[k] else None) for k in totals}
    for name in correct:
        metrics[f"{name}_acc"] = (
            correct[name] / counted[name] if counted[name] else None
        )
    if s_total_bits > 0:
        prec = s_tp / max(s_tp + s_fp, 1)
        rec = s_tp / max(s_tp + s_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics["strategy_acc"] = f1
        metrics["strategy_micro_f1"] = f1
        metrics["strategy_precision"] = prec
        metrics["strategy_recall"] = rec
        metrics["strategy_hamming_acc"] = s_correct_bits / s_total_bits
        metrics["strategy_exact_match"] = s_exact / max(s_n, 1)
    elif s_single_n > 0:
        metrics["strategy_acc"] = s_single_correct / s_single_n
    else:
        metrics["strategy_acc"] = None
    metrics["n_batches"] = n_batches_run
    metrics["n_examples"] = n_examples
    return metrics


@torch.no_grad()
def eval_stage1(model, loader, device, max_batches: int | None) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ["loss", "lm_loss", "emotion_loss"]}
    counts = {k: 0 for k in totals}
    correct = 0
    counted = 0
    n_examples = 0
    n_batches_run = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask", "labels", "emotion_ids"}
        }
        out = model(**batch)
        n_batches_run += 1
        n_examples += int(batch["input_ids"].size(0))
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
    metrics["n_batches"] = n_batches_run
    metrics["n_examples"] = n_examples
    return metrics


def load_stage3_bundle(cfg: dict, ckpt_dir: Path, device: torch.device) -> Stage3EmpathyModel:
    labels = json.loads((ckpt_dir / "labels.json").read_text(encoding="utf-8"))
    tok = build_tokenizer(cfg["model_name"])
    base = load_base_causal_lm(cfg["model_name"], dtype=cfg.get("dtype", "bf16"))
    # select_dual checkpoints: KO LoRA was trained on EN-merged base
    if labels.get("select_dual") or labels.get("select_bank"):
        en_dir = labels.get("stage1_lora_dir") or cfg.get("stage1_lora_dir")
        if not en_dir:
            en_dir = str(Path(cfg["stage1_dir"]) / "lora") if cfg.get("stage1_dir") else None
        if en_dir:
            en_path = ROOT / en_dir if not Path(en_dir).is_absolute() else Path(en_dir)
            base = PeftModel.from_pretrained(base, str(en_path)).merge_and_unload()
    if labels.get("madx"):
        en_dir = labels.get("stage1_lora_dir") or cfg.get("stage1_lora_dir")
        en_path = ROOT / en_dir if en_dir and not Path(en_dir).is_absolute() else Path(en_dir or ".")
        lm = PeftModel.from_pretrained(base, str(en_path), adapter_name="language")
        lm.load_adapter(str(resolve_stage3_lora_dir(ckpt_dir)), adapter_name="task")
        set_active_adapters(lm, ["language", "task"])
    else:
        lm = PeftModel.from_pretrained(base, str(ckpt_dir / "lora"))
    heads = torch.load(ckpt_dir / "heads.pt", map_location="cpu", weights_only=False)
    multilabel = bool(
        labels.get("strategy_multilabel", heads.get("strategy_multilabel", False))
    )
    # Detect deep strategy head from checkpoint tensor keys
    deep = any(k.startswith("0.") or k.startswith("3.") for k in heads["strategy_head"])
    model = Stage3EmpathyModel(
        lm,
        n_emotions=len(labels["emotion"]),
        n_strategies=len(labels["strategy"]),
        n_relations=len(labels["relation"]),
        strategy_multilabel=multilabel,
        deep_strategy_head=deep,
        two_pass_affect=bool(labels.get("two_pass_affect", False)),
    )
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
    base = load_base_causal_lm(cfg["model_name"], dtype=cfg.get("dtype", "bf16"))
    # If this LoRA came from select_dual/select_bank Stage3, bake Stage1 EN first then stack KO LoRA.
    labels_path = Path(lora_dir).parent / "labels.json"
    if labels_path.exists():
        meta = json.loads(labels_path.read_text(encoding="utf-8"))
        if meta.get("select_dual") or meta.get("select_bank"):
            en_rel = meta.get("stage1_lora_dir") or str(stage1_dir / "lora")
            en_path = ROOT / en_rel if not Path(en_rel).is_absolute() else Path(en_rel)
            base = PeftModel.from_pretrained(base, str(en_path)).merge_and_unload()
        elif meta.get("madx"):
            lora_dir = resolve_stage3_lora_dir(labels_path.parent)
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
            strategy_labels=labels.get("strategy") or None,
            relation_labels=labels.get("relation") or None,
        )
    has_strategy = bool(labels and labels.get("strategy"))
    has_relation = bool(labels and labels.get("relation"))
    if has_strategy or has_relation:
        kwargs["multitask"] = True
        kwargs["strategy_scope"] = labels.get("strategy_scope") or cfg.get(
            "strategy_scope", "utterance"
        )
    ds = EmpathyJsonlDataset(path, **kwargs)
    n_s = len(labels["strategy"]) if labels and labels.get("strategy") else None
    collator = EmpathyCollator(
        tok,
        max_length=int(cfg.get("max_length", 384)),
        n_strategies=n_s,
    )
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate full valid split (ignore config max_batches).",
    )
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    device = pick_device(str(cfg.get("device", "auto")))
    if args.full:
        max_batches = None
    elif args.max_batches is not None:
        max_batches = args.max_batches
    else:
        # YAML null → full split; omit → legacy default 50
        max_batches = cfg["max_batches"] if "max_batches" in cfg else 50

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
        "eval_protocol": {
            "max_batches": max_batches,
            "full_split": max_batches is None,
            "batch_size": int(cfg.get("batch_size", 2)),
        },
        "direction_i_ko": {},
        "direction_ii_en": {},
    }

    # ---- Direction I: KO affective accuracy ----
    print(
        f"=== Direction I: KO (AI Hub) max_batches={max_batches} "
        f"(None=full) ===",
        flush=True,
    )
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
            "n_dataset": len(ds),
            "n_eval": metrics.get("n_examples"),
            "n_batches": metrics.get("n_batches"),
            "metrics": metrics,
        }
        print(
            name,
            {
                k: metrics[k]
                for k in [
                    "emotion_acc",
                    "strategy_acc",
                    "relation_acc",
                    "lm_loss",
                    "n_examples",
                ]
            },
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ---- Direction II: return to EN ----
    print(
        f"=== Direction II: EN (EmpatheticDialogues) max_batches={max_batches} ===",
        flush=True,
    )
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
            "n_dataset": len(ds),
            "n_eval": metrics.get("n_examples"),
            "n_batches": metrics.get("n_batches"),
            "n_emotions": len(emo_labels),
            "metrics": metrics,
        }
        print(
            name,
            {k: metrics[k] for k in ["emotion_acc", "lm_loss", "loss", "n_examples"]},
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
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
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
