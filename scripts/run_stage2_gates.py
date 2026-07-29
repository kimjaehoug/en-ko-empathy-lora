#!/usr/bin/env python3
"""Stage2: EN→KO divergence / probe gates (share | relearn | suppress)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.empathy_data import EmpathyJsonlDataset
from src.models.encoding import (
    cosine_mean_similarity,
    encode_prompts,
    fit_linear_probe,
    load_stage1_bundle,
    pick_device,
    probe_accuracy,
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def gate_from_score(score: float, share_thr: float, suppress_thr: float) -> str:
    # high probe/similarity -> share; very low -> suppress harmful reuse; else relearn
    if score >= share_thr:
        return "share"
    if score <= suppress_thr:
        return "suppress"
    return "relearn"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage2_gates.yaml"))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    device = pick_device(str(cfg.get("device", "auto")))
    stage1_dir = ROOT / cfg["stage1_dir"]
    emotion_labels = json.loads((stage1_dir / "emotion_labels.json").read_text())

    model, tok = load_stage1_bundle(
        model_name=cfg["model_name"],
        stage1_dir=str(stage1_dir),
        n_emotions=len(emotion_labels),
        device=device,
    )

    max_en = int(cfg.get("max_en_samples", 256))
    max_ko = int(cfg.get("max_ko_samples", 256))
    max_len = int(cfg.get("max_length", 256))
    bs = int(cfg.get("batch_size", 4))

    en_ds = EmpathyJsonlDataset(ROOT / cfg["en_file"], max_history=int(cfg.get("max_history", 6)))
    ko_ds = EmpathyJsonlDataset(
        ROOT / cfg["ko_file"],
        max_history=int(cfg.get("max_history", 6)),
        multitask=True,
        lang_hint="ko",
    )
    # optional parallel cultural eval set
    koed_ds = None
    if cfg.get("koed_file"):
        koed_ds = EmpathyJsonlDataset(
            ROOT / cfg["koed_file"],
            max_history=int(cfg.get("max_history", 6)),
            lang_hint="ko",
        )

    en_rows = en_ds.rows[:max_en]
    ko_rows = ko_ds.rows[:max_ko]

    print(f"encoding EN={len(en_rows)} KO={len(ko_rows)} device={device}")
    en_x = encode_prompts(
        model, tok, [r["prompt"] for r in en_rows], device=device, max_length=max_len, batch_size=bs
    )
    ko_x = encode_prompts(
        model, tok, [r["prompt"] for r in ko_rows], device=device, max_length=max_len, batch_size=bs
    )

    domain_sim = cosine_mean_similarity(en_x, ko_x)

    # KO probes on frozen EN-stage encoder features
    ko_emotion = torch.tensor([r["emotion_id"] for r in ko_rows], dtype=torch.long)
    ko_strategy = torch.tensor([r["strategy_id"] for r in ko_rows], dtype=torch.long)
    ko_relation = torch.tensor([r["relation_id"] for r in ko_rows], dtype=torch.long)

    # remap KO emotion ids already in dataset space
    n_emo = len(ko_ds.emotion_labels)
    n_str = len(ko_ds.strategy_labels)
    n_rel = len(ko_ds.relation_labels)

    # train/valid split inside KO sample for probe generalization
    n = len(ko_rows)
    n_train = max(int(0.8 * n), 1)
    x_tr, x_te = ko_x[:n_train], ko_x[n_train:]
    metrics = {"domain_cosine": domain_sim}

    def run_probe(name, y_all, n_classes):
        y_tr, y_te = y_all[:n_train], y_all[n_train:]
        probe, train_acc = fit_linear_probe(
            x_tr, y_tr, n_classes=n_classes, steps=int(cfg.get("probe_steps", 250))
        )
        te_acc = probe_accuracy(probe, x_te, y_te) if len(x_te) else train_acc
        metrics[name] = {"train_acc": train_acc, "valid_acc": te_acc, "n_classes": n_classes}
        return te_acc if te_acc == te_acc else train_acc

    affect_score = run_probe("affect_probe", ko_emotion, n_emo)
    strategy_score = run_probe("strategy_probe", ko_strategy, n_str)
    relation_score = run_probe("relation_probe", ko_relation, n_rel)

    # parallel KoED surface similarity (culture signal)
    culture_score = domain_sim
    if koed_ds is not None:
        # encode KO prompts; also encode EN situation+dialogue text_en variant if present
        # For processed koed, prompt is KO; approximate culture gap via lower domain_sim
        # and optional EN/KO situation embedding gap from raw fields in prompts
        koed_rows = koed_ds.rows[: min(len(koed_ds), int(cfg.get("max_koed_samples", 128)))]
        koed_x = encode_prompts(
            model,
            tok,
            [r["prompt"] for r in koed_rows],
            device=device,
            max_length=max_len,
            batch_size=bs,
        )
        culture_score = cosine_mean_similarity(en_x[: len(koed_x)], koed_x)
        metrics["koed_parallel_cosine"] = culture_score

    share_thr = float(cfg.get("share_threshold", 0.35))
    suppress_thr = float(cfg.get("suppress_threshold", 0.12))

    # Combine domain similarity into affect/culture decisions
    affect_gate_score = 0.5 * affect_score + 0.5 * max(0.0, domain_sim)
    culture_gate_score = culture_score

    gates = {
        "affect": {
            "score": affect_gate_score,
            "decision": gate_from_score(affect_gate_score, share_thr, suppress_thr),
        },
        "strategy": {
            "score": strategy_score,
            "decision": gate_from_score(strategy_score, share_thr, suppress_thr),
        },
        "relation": {
            "score": relation_score,
            "decision": gate_from_score(relation_score, share_thr, suppress_thr),
        },
        "culture": {
            "score": culture_gate_score,
            "decision": gate_from_score(culture_gate_score, share_thr, suppress_thr),
        },
    }

    # Hard prior: strategy/relation labels absent in EN -> prefer relearn unless probe is strong
    if strategy_score < share_thr:
        gates["strategy"]["decision"] = "relearn"
    if relation_score < share_thr:
        gates["relation"]["decision"] = "relearn"

    out = {
        "model_name": cfg["model_name"],
        "stage1_dir": str(stage1_dir),
        "thresholds": {"share": share_thr, "suppress": suppress_thr},
        "metrics": metrics,
        "gates": gates,
        "label_spaces": {
            "ko_emotion": ko_ds.emotion_labels,
            "ko_strategy": ko_ds.strategy_labels,
            "ko_relation": ko_ds.relation_labels,
        },
        "policy": {
            "share": "init KO LoRA from EN LoRA; small LR",
            "relearn": "re-init KO LoRA; normal LR",
            "suppress": "re-init KO LoRA + EN-style hard-negative / lower share weight",
        },
    }

    out_dir = ROOT / cfg.get("output_dir", "outputs/stage2_gates")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gates.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
