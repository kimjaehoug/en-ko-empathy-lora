#!/usr/bin/env python3
"""Summarize Dir I/II across Qwen / Llama / EXAONE for backbone×SELECT interaction.

Reads:
  outputs/qwen35_9b/eval_direction_i_ii/report.json
  outputs/llama31_8b/eval_direction_i_ii/report.json
  outputs/exaone30_7p8b/eval_direction_i_ii/report.json

Writes:
  outputs/cross_backbone/summary.json
  outputs/cross_backbone/summary.md

ΔAcc definitions (Direction I):
  Δ_blind  = SELECT − BlindShare
  Δ_scratch = SELECT − KO-scratch
If |Δ| varies a lot across backbones, method gain interacts with backbone
(contamination / non-additivity). If Δ is stable, SELECT is relatively backbone-robust.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "Qwen3.5-9B": ROOT / "outputs/qwen35_9b/eval_direction_i_ii/report.json",
    "Llama-3.1-8B": ROOT / "outputs/llama31_8b/eval_direction_i_ii/report.json",
    "EXAONE-3.0-7.8B": ROOT / "outputs/exaone30_7p8b/eval_direction_i_ii/report.json",
}


def _acc(blob: dict, system: str, key: str):
    m = (blob.get("direction_i_ko") or {}).get(system, {}).get("metrics") or {}
    return m.get(key)


def _en_acc(blob: dict, name: str):
    m = (blob.get("direction_ii_en") or {}).get(name, {}).get("metrics") or {}
    return m.get("emotion_acc")


def main() -> None:
    rows = []
    for name, path in FAMILIES.items():
        if not path.exists():
            rows.append({"backbone": name, "status": "missing", "path": str(path)})
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        sel_a = _acc(blob, "select", "emotion_acc")
        sel_s = _acc(blob, "select", "strategy_acc")
        sel_r = _acc(blob, "select", "relation_acc")
        bli_a = _acc(blob, "blind_share", "emotion_acc")
        bli_s = _acc(blob, "blind_share", "strategy_acc")
        bli_r = _acc(blob, "blind_share", "relation_acc")
        scr_a = _acc(blob, "ko_scratch", "emotion_acc")
        scr_s = _acc(blob, "ko_scratch", "strategy_acc")
        scr_r = _acc(blob, "ko_scratch", "relation_acc")

        def sub(a, b):
            if a is None or b is None:
                return None
            return a - b

        row = {
            "backbone": name,
            "status": "ok",
            "dir_i": {
                "select": {"A": sel_a, "S": sel_s, "R": sel_r},
                "blind": {"A": bli_a, "S": bli_s, "R": bli_r},
                "scratch": {"A": scr_a, "S": scr_s, "R": scr_r},
                "delta_select_minus_blind": {
                    "A": sub(sel_a, bli_a),
                    "S": sub(sel_s, bli_s),
                    "R": sub(sel_r, bli_r),
                },
                "delta_select_minus_scratch": {
                    "A": sub(sel_a, scr_a),
                    "S": sub(sel_s, scr_s),
                    "R": sub(sel_r, scr_r),
                },
            },
            "dir_ii_emotion_acc": {
                "en_before": _en_acc(blob, "en_before_ko"),
                "after_select": _en_acc(blob, "after_select"),
                "after_blind": _en_acc(blob, "after_blind_share"),
                "after_scratch": _en_acc(blob, "after_ko_scratch"),
            },
        }
        # Dir II forgetting: after − before
        eb = row["dir_ii_emotion_acc"]["en_before"]
        for k in ("after_select", "after_blind", "after_scratch"):
            v = row["dir_ii_emotion_acc"][k]
            row["dir_ii_emotion_acc"][f"delta_{k}"] = (
                None if v is None or eb is None else v - eb
            )
        rows.append(row)

    out_dir = ROOT / "outputs/cross_backbone"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Cross-backbone SELECT summary",
        "",
        "Protocol: same Stage1→2→3 SELECT / Blind / Scratch on each backbone.",
        "",
        "## Direction I — absolute Acc",
        "",
        "| Backbone | SELECT A/S/R | Blind A/S/R | Scratch A/S/R |",
        "|----------|--------------|-------------|---------------|",
    ]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['backbone']} | missing | — | — |")
            continue
        s, b, c = r["dir_i"]["select"], r["dir_i"]["blind"], r["dir_i"]["scratch"]

        def fmt(d):
            return "/".join(
                f"{100 * d[k]:.1f}" if d[k] is not None else "n/a" for k in ("A", "S", "R")
            )

        lines.append(f"| {r['backbone']} | {fmt(s)} | {fmt(b)} | {fmt(c)} |")

    lines += [
        "",
        "## Direction I — ΔAcc (SELECT − Blind / SELECT − Scratch)",
        "",
        "| Backbone | ΔBlind A/S/R | ΔScratch A/S/R |",
        "|----------|--------------|----------------|",
    ]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['backbone']} | missing | — |")
            continue
        db = r["dir_i"]["delta_select_minus_blind"]
        ds = r["dir_i"]["delta_select_minus_scratch"]

        def fmt(d):
            parts = []
            for k in ("A", "S", "R"):
                v = d[k]
                parts.append(f"{100 * v:+.1f}" if v is not None else "n/a")
            return "/".join(parts)

        lines.append(f"| {r['backbone']} | {fmt(db)} | {fmt(ds)} |")

    lines += [
        "",
        "## Interpretation guide",
        "",
        "- If ΔBlind/ΔScratch **sign and magnitude are similar** across backbones → SELECT gain is relatively backbone-robust.",
        "- If Δ **flips or shrinks** on EXAONE/Llama only → method×backbone interaction (cannot attribute gain to SELECT alone).",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
