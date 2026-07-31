#!/usr/bin/env python3
"""Rebuild Stage2 gates from an existing gates.json metrics blob (no re-encode).

Uses probe-heavy affect scoring so A can share while S/R stay relearn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def gate_from_score(score: float, share_thr: float, suppress_thr: float) -> str:
    if score >= share_thr:
        return "share"
    if score <= suppress_thr:
        return "suppress"
    return "relearn"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="outputs/qwen35_9b/stage2_gates/gates.json")
    p.add_argument("--dst", default="outputs/qwen35_9b_v3/stage2_gates/gates.json")
    p.add_argument("--share_threshold", type=float, default=0.55)
    p.add_argument("--suppress_threshold", type=float, default=0.15)
    p.add_argument("--sr_share_threshold", type=float, default=0.75)
    p.add_argument("--affect_probe_weight", type=float, default=0.9)
    p.add_argument("--force_sr_relearn", action="store_true", default=True)
    args = p.parse_args()

    src = Path(args.src)
    blob = json.loads(src.read_text(encoding="utf-8"))
    metrics = blob["metrics"]
    affect_probe = float(metrics["affect_probe"]["valid_acc"])
    strategy_score = float(metrics["strategy_probe"]["valid_acc"])
    relation_score = float(metrics["relation_probe"]["valid_acc"])
    domain = float(metrics.get("domain_cosine", 0.0))
    culture = float(metrics.get("koed_parallel_cosine", domain))

    w = args.affect_probe_weight
    affect_score = w * affect_probe + (1.0 - w) * max(0.0, domain)

    gates = {
        "affect": {
            "score": affect_score,
            "decision": gate_from_score(
                affect_score, args.share_threshold, args.suppress_threshold
            ),
            "note": f"probe={affect_probe:.4f} domain={domain:.4f} w={w}",
        },
        "strategy": {
            "score": strategy_score,
            "decision": gate_from_score(
                strategy_score, args.sr_share_threshold, args.suppress_threshold
            ),
        },
        "relation": {
            "score": relation_score,
            "decision": gate_from_score(
                relation_score, args.sr_share_threshold, args.suppress_threshold
            ),
        },
        "culture": {
            "score": culture,
            "decision": gate_from_score(
                culture, args.share_threshold, args.suppress_threshold
            ),
        },
    }
    if args.force_sr_relearn:
        if strategy_score < args.sr_share_threshold:
            gates["strategy"]["decision"] = "relearn"
            gates["strategy"]["prior"] = "en_missing_sr_labels"
        if relation_score < args.sr_share_threshold:
            gates["relation"]["decision"] = "relearn"
            gates["relation"]["prior"] = "en_missing_sr_labels"

    decisions = [v["decision"] for v in gates.values()]
    if all(d == "share" for d in decisions):
        raise SystemExit(f"all-share collapse: {gates}")

    out = dict(blob)
    out["thresholds"] = {
        "share": args.share_threshold,
        "suppress": args.suppress_threshold,
        "sr_share": args.sr_share_threshold,
        "force_sr_relearn": args.force_sr_relearn,
        "affect_probe_weight": w,
        "rebuilt_from": str(src),
    }
    out["gates"] = gates
    out["policy_note"] = (
        "v3: affect uses probe-heavy score; S/R forced relearn under EN-missing prior"
    )

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("gates", {k: v["decision"] for k, v in gates.items()})
    print("scores", {k: round(v["score"], 4) for k, v in gates.items()})
    print("wrote", dst)


if __name__ == "__main__":
    main()
