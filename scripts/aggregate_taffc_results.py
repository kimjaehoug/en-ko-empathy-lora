#!/usr/bin/env python3
"""Aggregate TAFFC multi-seed runs: mean±std, paired tests, LaTeX snippets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
from src.eval.metrics import aggregate_seeds, paired_ttest  # noqa: E402


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def dir_i_avg(metrics: dict) -> float | None:
    m = metrics or {}
    a, s, r = m.get("emotion_acc"), m.get("strategy_acc"), m.get("relation_acc")
    if a is None or s is None or r is None:
        return None
    return 100 * (a + s + r) / 3


def collect_seed_metrics(base: Path, run_id: str, seeds: list[int]) -> dict:
    out = {"A": [], "S": [], "R": [], "Avg": [], "DirII": []}
    for seed in seeds:
        ckpt = base / run_id / f"s{seed}"
        # Prefer extended eval report; fall back to train final_eval
        rep = load_report(ckpt / "eval/report.json") or load_report(
            base.parent / "eval" / f"{run_id}_s{seed}.json"
        )
        if rep and "direction_i_ko" in rep:
            m = (rep["direction_i_ko"].get(run_id) or {}).get("metrics") or {}
        else:
            meta = load_report(ckpt / "run_meta.json")
            m = (meta or {}).get("final_eval") or {}
        if m.get("emotion_acc") is not None:
            out["A"].append(100 * m["emotion_acc"])
            out["S"].append(100 * m.get("strategy_acc") or m.get("strategy_micro_f1", 0))
            out["R"].append(100 * m["relation_acc"])
            avg = dir_i_avg(m)
            if avg is not None:
                out["Avg"].append(avg)
        rep2 = load_report(ROOT / "outputs/taffc/eval/report.json")
        if rep2:
            d2 = (rep2.get("direction_ii_en") or {}).get(f"after_{run_id}", {})
            acc = (d2.get("metrics") or {}).get("emotion_acc")
            if acc is not None:
                out["DirII"].append(100 * acc)
    return {k: aggregate_seeds(v) for k, v in out.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(ROOT / "configs/taffc/ablation_matrix.yaml"))
    parser.add_argument("--out", default=str(ROOT / "outputs/taffc/aggregate.json"))
    args = parser.parse_args()
    matrix = yaml.safe_load(Path(args.matrix).read_text(encoding="utf-8"))
    seeds = matrix.get("seeds", [42, 123, 456])

    summary = {"seeds": seeds, "main": {}, "paired_tests": {}}
    base_qwen = ROOT / "outputs/taffc/qwen"

    for run in matrix.get("main_runs", []):
        rid = run["id"]
        summary["main"][rid] = collect_seed_metrics(base_qwen, rid, seeds)

    # Paired: F16 vs B16 on Avg (per-seed)
    f_avg = []
    b_avg = []
    for seed in seeds:
        for rid, bucket in [("F16", f_avg), ("B16", b_avg)]:
            ckpt = base_qwen / rid / f"s{seed}"
            meta = load_report(ckpt / "run_meta.json")
            m = (meta or {}).get("final_eval") or {}
            avg = dir_i_avg(m)
            if avg is not None:
                bucket.append(avg)
    if len(f_avg) == len(b_avg) and len(f_avg) >= 2:
        summary["paired_tests"]["F16_vs_B16_Avg"] = paired_ttest(f_avg, b_avg)

    # v6 single-seed reference
    v6 = load_report(ROOT / "outputs/qwen35_9b_v6/summary.json")
    if v6:
        summary["v6_reference"] = v6

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    # Markdown table
    md = ["# TAFFC aggregate (multi-seed)\n\n", "| System | A | S | R | Avg |\n|---|---:|---:|---:|---:|\n"]
    for rid, stats in summary["main"].items():
        def cell(k):
            s = stats.get(k, {})
            if not s.get("mean"):
                return "—"
            return f"{s['mean']:.1f}±{s.get('std', 0):.1f}"

        md.append(f"| {rid} | {cell('A')} | {cell('S')} | {cell('R')} | {cell('Avg')} |\n")
    (out_path.parent / "aggregate.md").write_text("".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
