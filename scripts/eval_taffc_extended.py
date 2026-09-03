#!/usr/bin/env python3
"""TAFFC extended evaluation: Dir I/II, KoED, confusion, per-class F1, error analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_direction_i_ii import (  # noqa: E402
    eval_stage1,
    eval_stage3,
    load_config,
    load_stage1_for_en,
    load_stage3_bundle,
    make_loader,
)
from src.data.empathy_data import EmpathyJsonlDataset, load_jsonl  # noqa: E402
from src.eval.koed_mapping import (  # noqa: E402
    ED_TO_SHARED,
    KOED_SHARED_EMOTIONS,
    koed_multilabel_hits,
    koed_primary_emotion,
)
from src.eval.metrics import (  # noqa: E402
    binary_multilabel_metrics,
    compare_pred_vectors,
    error_buckets,
    per_class_report,
)
from src.models.encoding import pick_device  # noqa: E402


@torch.no_grad()
def collect_stage3_predictions(model, loader, device, max_batches) -> dict:
    model.eval()
    emo_pred, emo_gold = [], []
    rel_pred, rel_gold = [], []
    strat_pred, strat_gold = [], []
    example_ids = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bsz = batch["input_ids"].size(0)
        for j in range(bsz):
            example_ids.append(f"batch{i}_row{j}")

        for name, logits_key, ids_key in [
            ("emotion", "emotion_logits", "emotion_ids"),
            ("relation", "relation_logits", "relation_ids"),
        ]:
            ids = batch[ids_key]
            valid = ids >= 0
            if valid.any():
                pred = out[logits_key][valid].argmax(dim=-1).cpu().tolist()
                gold = ids[valid].cpu().tolist()
                if name == "emotion":
                    emo_pred.extend(pred)
                    emo_gold.extend(gold)
                else:
                    rel_pred.extend(pred)
                    rel_gold.extend(gold)

        if "strategy_multihot" in batch:
            target = batch["strategy_multihot"]
            pred = (out["strategy_logits"].sigmoid() >= 0.5).float()
            row_has = target.sum(dim=-1) > 0
            if row_has.any():
                strat_pred.extend(pred[row_has].cpu().int().tolist())
                strat_gold.extend(target[row_has].cpu().int().tolist())

    return {
        "emotion": {"pred": emo_pred, "gold": emo_gold},
        "relation": {"pred": rel_pred, "gold": rel_gold},
        "strategy": {"pred": strat_pred, "gold": strat_gold},
        "example_ids": example_ids,
    }


@torch.no_grad()
def eval_koed_ed_emotion(model, tok, ed_labels: list[str], cfg: dict, device, max_batches) -> dict:
    """KoED held-out: Korean dialogue → ED 32-class head; report 8-class shared subset."""
    shared_ids = {e: i for i, e in enumerate(KOED_SHARED_EMOTIONS)}
    ed2id = {l: i for i, l in enumerate(ed_labels)}

    from src.data.empathy_data import EmpathyCollator
    from torch.utils.data import DataLoader

    koed_path = ROOT / cfg.get("koed_file", "data/processed/koed_test.jsonl")
    ds = EmpathyJsonlDataset(
        koed_path,
        max_history=int(cfg.get("max_history", 8)),
        lang_hint="ko",
        emotion_labels=ed_labels,
        multitask=False,
    )
    ds.rows = [r for r in ds.rows if r.get("emotion_id", -1) >= 0]
    collator = EmpathyCollator(tok, max_length=int(cfg.get("max_length", 512)))
    loader = DataLoader(
        ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=False, collate_fn=collator
    )

    preds_full, golds_full = [], []
    preds_shared, golds_shared = [], []
    hits_any = n_eval = 0
    raw_rows = load_jsonl(koed_path)
    row_map = {r.get("id"): r for r in raw_rows}

    model.eval()
    offset = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in {"input_ids", "attention_mask", "labels", "emotion_ids"}
        }
        out = model(**batch)
        ids = batch["emotion_ids"]
        valid = ids >= 0
        if not valid.any():
            continue
        pred_ids = out["emotion_logits"][valid].argmax(dim=-1).cpu().tolist()
        gold_ids = ids[valid].cpu().tolist()
        for p_id, g_id in zip(pred_ids, gold_ids):
            n_eval += 1
            preds_full.append(p_id)
            golds_full.append(g_id)
            p_lab = ed_labels[p_id]
            g_lab = ed_labels[g_id]
            p_shared = ED_TO_SHARED.get(p_lab, p_lab if p_lab in shared_ids else None)
            g_shared = ED_TO_SHARED.get(g_lab, g_lab if g_lab in shared_ids else None)
            if p_shared in shared_ids and g_shared in shared_ids:
                preds_shared.append(shared_ids[p_shared])
                golds_shared.append(shared_ids[g_shared])
            ex = ds.rows[offset] if offset < len(ds.rows) else {}
            row = row_map.get(ex.get("id")) or {}
            aff = (row.get("axes") or {}).get("A_affect")
            gold_set = koed_multilabel_hits(aff)
            pred_mapped = ED_TO_SHARED.get(p_lab, p_lab)
            if pred_mapped in gold_set:
                hits_any += 1
            offset += 1

    report32 = per_class_report(golds_full, preds_full, ed_labels) if golds_full else {}
    report8 = (
        per_class_report(golds_shared, preds_shared, KOED_SHARED_EMOTIONS)
        if golds_shared
        else {}
    )
    return {
        "n_eval": n_eval,
        "full_32class": {
            "accuracy": report32.get("accuracy"),
            "macro_f1": report32.get("macro_f1"),
            "per_class": report32.get("per_class"),
            "confusion_matrix": report32.get("confusion_matrix"),
            "error_buckets": error_buckets(golds_full, preds_full, ed_labels),
        },
        "shared_8class": {
            "accuracy": report8.get("accuracy"),
            "macro_f1": report8.get("macro_f1"),
            "per_class": report8.get("per_class"),
            "confusion_matrix": report8.get("confusion_matrix"),
            "error_buckets": error_buckets(golds_shared, preds_shared, KOED_SHARED_EMOTIONS),
        },
        "multilabel_any_hit_rate": hits_any / n_eval if n_eval else None,
    }


def compare_systems_error(
    preds_a: dict, preds_b: dict, labels: list[str], name_a: str, name_b: str
) -> dict:
    emo_a, emo_b = preds_a["emotion"]["pred"], preds_b["emotion"]["pred"]
    gold = preds_a["emotion"]["gold"]
    if len(emo_a) != len(emo_b) or len(emo_a) != len(gold):
        return {"error": "length mismatch"}
    a_wins = b_wins = both_wrong = 0
    by_class = {l: {"a_wins": 0, "b_wins": 0, "n": 0} for l in labels}
    for p_a, p_b, g in zip(emo_a, emo_b, gold):
        la, lb, lg = labels[p_a], labels[p_b], labels[g]
        by_class[lg]["n"] += 1
        ca, cb = p_a == g, p_b == g
        if ca and not cb:
            a_wins += 1
            by_class[lg]["a_wins"] += 1
        elif cb and not ca:
            b_wins += 1
            by_class[lg]["b_wins"] += 1
        elif not ca and not cb:
            both_wrong += 1
    return {
        "compare": f"{name_a}_vs_{name_b}",
        "a_wins": a_wins,
        "b_wins": b_wins,
        "both_wrong": both_wrong,
        "n": len(gold),
        "pred_match_rate": compare_pred_vectors(emo_a, emo_b)["match_rate"],
        "per_class_a_wins": by_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/taffc/eval_taffc.yaml"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--dump_predictions", action="store_true")
    parser.add_argument("--error_analysis", action="store_true")
    parser.add_argument("--koed", action="store_true", help="Run KoED held-out eval")
    parser.add_argument("--reference", default="F16", help="System A for error compare")
    parser.add_argument("--baseline", default="B16", help="System B for error compare")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    device = pick_device(str(cfg.get("device", "auto")))
    max_batches = None if args.full else (args.max_batches if args.max_batches is not None else cfg.get("max_batches"))

    out_dir = ROOT / cfg.get("output_dir", "outputs/taffc/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    report = {
        "eval_protocol": {"max_batches": max_batches, "full": max_batches is None},
        "direction_i_ko": {},
        "direction_ii_en": {},
        "koed": {},
        "error_analysis": {},
    }

    stage1_dir = ROOT / cfg["stage1_dir"]
    collected = {}

    # Dir I
    for name, rel in cfg.get("ko_checkpoints", {}).items():
        ckpt = ROOT / rel
        if not (ckpt / "lora").exists():
            report["direction_i_ko"][name] = {"error": f"missing {ckpt}"}
            continue
        model, tok, labels = load_stage3_bundle(cfg, ckpt, device)
        valid_rel = (
            cfg.get("ko_checkpoint_valid_files", {}).get(name) or cfg["ko_valid_file"]
        )
        loader, ds = make_loader(
            ROOT / valid_rel, tok, cfg, lang_hint="ko", labels=labels
        )
        metrics = eval_stage3(model, loader, device, max_batches)
        emo_labels = labels["emotion"]
        if args.dump_predictions or args.error_analysis:
            collected[name] = collect_stage3_predictions(model, loader, device, max_batches)
            emo = collected[name]["emotion"]
            pc = per_class_report(emo["gold"], emo["pred"], emo_labels)
            strat = collected[name]["strategy"]
            strat_m = (
                binary_multilabel_metrics(strat["gold"], strat["pred"])
                if strat["gold"]
                else {}
            )
            rel = collected[name]["relation"]
            rel_pc = (
                per_class_report(rel["gold"], rel["pred"], labels["relation"])
                if labels.get("relation") and rel["gold"]
                else {}
            )
            detail = {
                "emotion": {
                    "per_class": pc["per_class"],
                    "confusion_matrix": pc["confusion_matrix"],
                    "macro_f1": pc["macro_f1"],
                    "accuracy": pc["accuracy"],
                    "error_buckets": error_buckets(emo["gold"], emo["pred"], emo_labels),
                },
            }
            if rel_pc:
                detail["relation"] = {
                    "per_class": rel_pc["per_class"],
                    "confusion_matrix": rel_pc["confusion_matrix"],
                }
            if strat_m:
                detail["strategy"] = strat_m
            (pred_dir / f"{name}_analysis.json").write_text(
                json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if args.dump_predictions:
                (pred_dir / f"{name}_emotion_preds.json").write_text(
                    json.dumps(collected[name]["emotion"], ensure_ascii=False), encoding="utf-8"
                )

        report["direction_i_ko"][name] = {
            "checkpoint": str(ckpt),
            "n_dataset": len(ds),
            "metrics": metrics,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.error_analysis and args.reference in collected and args.baseline in collected:
        ref_ckpt = ROOT / cfg["ko_checkpoints"][args.reference]
        ref_labels = json.loads((ref_ckpt / "labels.json").read_text(encoding="utf-8"))["emotion"]
        report["error_analysis"] = compare_systems_error(
            collected[args.reference],
            collected[args.baseline],
            ref_labels,
            args.reference,
            args.baseline,
        )

    # Dir II
    for name, rel in cfg.get("en_lora_variants", {}).items():
        lora_path = ROOT / rel
        if not lora_path.exists():
            continue
        model, tok, emo_labels = load_stage1_for_en(cfg, stage1_dir, lora_path, device)
        loader, ds = make_loader(
            ROOT / cfg["en_valid_file"], tok, cfg, labels={"emotion": emo_labels}
        )
        metrics = eval_stage1(model, loader, device, max_batches)
        report["direction_ii_en"][name] = {"metrics": metrics, "n_dataset": len(ds)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # KoED
    if args.koed:
        ed_labels = json.loads((stage1_dir / "emotion_labels.json").read_text())
        for name, rel in cfg.get("en_lora_variants", {}).items():
            if name == "en_before_ko":
                continue
            lora_path = ROOT / rel
            if not lora_path.exists():
                continue
            model, tok, _ = load_stage1_for_en(cfg, stage1_dir, lora_path, device)
            report["koed"][name] = eval_koed_ed_emotion(
                model, tok, ed_labels, cfg, device, max_batches
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    before = (report["direction_ii_en"].get("en_before_ko") or {}).get("metrics")
    if before and before.get("emotion_acc") is not None:
        for name, blob in report["direction_ii_en"].items():
            m = blob.get("metrics") or {}
            if m.get("emotion_acc") is not None:
                blob["delta_vs_en_before"] = m["emotion_acc"] - before["emotion_acc"]

    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
