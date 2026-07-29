#!/usr/bin/env python3
"""Compute split statistics for processed EN/KO empathy corpora."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "split_stats.json"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize(path: Path) -> dict:
    n = 0
    n_utt = 0
    n_listener = 0
    langs = Counter()
    A = Counter()
    S = Counter()
    R = Counter()
    has_S = 0
    has_R = 0
    utt_lens = []
    dlg_lens = []

    for row in iter_jsonl(path):
        n += 1
        langs[row.get("lang") or "?"] += 1
        axes = row.get("axes") or {}
        a = axes.get("A_affect")
        if isinstance(a, list):
            for x in a:
                A[str(x)] += 1
        elif a is not None:
            A[str(a)] += 1
        s = axes.get("S_strategy") or []
        if s:
            has_S += 1
            for x in s:
                S[str(x)] += 1
        r = axes.get("R_relation")
        if r:
            has_R += 1
            R[str(r)] += 1
        dlg = row.get("dialogue") or []
        dlg_lens.append(len(dlg))
        for u in dlg:
            n_utt += 1
            text = u.get("text") or ""
            utt_lens.append(len(text))
            if u.get("role") == "listener":
                n_listener += 1

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        "n_dialogues": n,
        "n_utterances": n_utt,
        "n_listener_turns": n_listener,
        "langs": dict(langs),
        "coverage": {
            "A_affect": round(sum(A.values()) / max(n, 1), 4),
            "S_strategy": round(has_S / max(n, 1), 4),
            "R_relation": round(has_R / max(n, 1), 4),
        },
        "avg_dialogue_turns": avg(dlg_lens),
        "avg_utt_chars": avg(utt_lens),
        "top_A": A.most_common(12),
        "top_S": S.most_common(12),
        "top_R": R.most_common(12),
    }


def main() -> None:
    files = sorted(PROC.glob("*.jsonl"))
    report: dict = {"files": {}, "by_source_split": {}, "stage_views": {}}
    by_source = defaultdict(dict)

    for path in files:
        key = path.stem  # e.g. ed_train
        stats = summarize(path)
        report["files"][key] = stats
        if "_" in key:
            source, split = key.rsplit("_", 1)
            by_source[source][split] = {
                "n_dialogues": stats["n_dialogues"],
                "coverage": stats["coverage"],
            }
        print(
            f"{key:18s} n={stats['n_dialogues']:6d} "
            f"A={stats['coverage']['A_affect']:.2f} "
            f"S={stats['coverage']['S_strategy']:.2f} "
            f"R={stats['coverage']['R_relation']:.2f}"
        )

    report["by_source_split"] = dict(by_source)

    # Recommended stage views for training
    report["stage_views"] = {
        "stage1_en": {
            "train": "ed_train.jsonl",
            "valid": "ed_valid.jsonl",
            "test": "ed_test.jsonl",
            "supervised_axes": ["A_affect", "C_cognition"],
            "n_train": report["files"].get("ed_train", {}).get("n_dialogues"),
            "n_valid": report["files"].get("ed_valid", {}).get("n_dialogues"),
        },
        "stage3_ko_adapt": {
            "train": "aihub_train.jsonl",
            "valid": "aihub_valid.jsonl",
            "eval_cultural": "koed_test.jsonl",
            "baseline_translated": "kor_ed_train.jsonl",
            "supervised_axes": ["A_affect", "S_strategy", "R_relation", "C_cognition"],
            "n_train": report["files"].get("aihub_train", {}).get("n_dialogues"),
            "n_valid": report["files"].get("aihub_valid", {}).get("n_dialogues"),
        },
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
