#!/usr/bin/env python3
"""Fill paper tables from v6 Factor-Bank matrix results."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/qwen35_9b_v6/summary.json"
REPORT = ROOT / "outputs/qwen35_9b_v6/eval_direction_i_ii/report.json"
TEX = ROOT / "Computer_Society_LaTeX_template/factor_lora_select.tex"
MD = ROOT / "paper/RESULTS_QWEN35.md"


def fmt_row(rid: str, row: dict, winner: str) -> str:
    a, s, rel, avg = row["A"], row["S"], row["R"], row["Avg"]
    if rid == winner:
        return f"| **{rid}** | {a:.1f} | {s:.1f} | {rel:.1f} | **{avg:.1f}** |"
    return f"| {rid} | {a:.1f} | {s:.1f} | {rel:.1f} | {avg:.1f} |"


def main() -> None:
    if not SUMMARY.exists() or not REPORT.exists():
        raise SystemExit(f"missing {SUMMARY} or {REPORT}")
    s = json.loads(SUMMARY.read_text())
    r = json.loads(REPORT.read_text())
    by = {row["id"]: row for row in s["dir_i"]}
    rank = s["avg_rank"]
    winner = rank[0]
    win_avg = by[winner]["Avg"]
    b16 = by.get("B16", {})
    f32 = by.get("F32", {})
    b32 = by.get("B32", {})

    claim_avg = winner.startswith("F") and win_avg > b16.get("Avg", 0)
    claim_axes = all(
        max(s["dir_i"], key=lambda x: x[k])["id"].startswith("F") for k in ("A", "S", "R")
    )

    lines = [
        "# SELENE Factor-Bank v6 (Qwen3.5-9B, full valid, utterance S)\n\n",
        "| ID | A | S F1 | R | Avg | n |\n|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in s["dir_i"]:
        lines.append(
            f"| {row['id']} | {row['A']:.2f} | {row['S']:.2f} | {row['R']:.2f} "
            f"| {row['Avg']:.2f} | {row['n']} |\n"
        )
    lines.append(f"\n**Avg rank:** {' > '.join(rank)}\n")
    lines.append(f"**Avg winner:** {winner} ({win_avg:.2f})\n\n## Per-axis winners\n")
    for ax, key in [("A", "A"), ("S", "S"), ("R", "R")]:
        best = max(s["dir_i"], key=lambda x: x[key])
        lines.append(f"- {ax}: **{best['id']}** {best[key]:.2f}\n")

    lines.append("\n## Claims\n")
    if claim_avg:
        lines.append(
            f"- Dir I **Avg 1등**: {winner} ({win_avg:.2f}) > B16 ({b16.get('Avg', 0):.2f})\n"
        )
    else:
        lines.append("- Dir I Avg 1등: **미달성**\n")
    if claim_axes:
        lines.append("- A/S/R **각각 1등**: 달성\n")
    else:
        lines.append("- A/S/R 각각 1등: 부분/미달성\n")
    if f32 and b32 and f32["Avg"] > b16.get("Avg", 0) and f32["Avg"] <= b32.get("Avg", 999):
        lines.append("- F32>B16 but F32≤B32 → 용량 효과 가능\n")

    b = r["direction_ii_en"]["en_before_ko"]["metrics"]["emotion_acc"]
    lines.append("\n## Dir II (ED emotion Acc)\n")
    lines.append(f"| System | Acc | Δ vs {100*b:.1f}% |\n|---|---:|---:|\n")
    for k, label in [
        ("after_F16", "After F16"),
        ("after_F32", "After F32"),
        ("after_B16", "After B16"),
        ("after_B32", "After B32"),
        ("after_S16", "After S16"),
    ]:
        m = (r["direction_ii_en"].get(k) or {}).get("metrics") or {}
        if not m:
            continue
        a = m["emotion_acc"]
        lines.append(f"| {label} | {100*a:.1f} | {100*(a-b):+.1f} |\n")

    MD.write_text("".join(lines))
    print(f"wrote {MD}")

    if not TEX.exists():
        return
    tex = TEX.read_text()
    table_rows = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"System & A Acc$\uparrow$ & S F1$\uparrow$ & R Acc$\uparrow$ & Avg$\uparrow$ \\",
        r"\midrule",
    ]
    labels = {
        "S16": "KO-scratch (S16)",
        "B16": "Blind share (B16)",
        "B32": "Blind share (B32)",
        "F16": r"SELENE bank (F16)",
        "F32": r"SELENE bank (F32)",
    }
    for rid in ["S16", "B16", "F16", "B32", "F32"]:
        row = by.get(rid)
        if not row:
            continue
        name = labels.get(rid, rid)
        a, s_, rel, avg = row["A"], row["S"], row["R"], row["Avg"]
        if rid == winner:
            table_rows.append(
                f"\\textbf{{{name}}} & {a:.1f} & {s_:.1f} & {rel:.1f} & \\textbf{{{avg:.1f}}} \\\\"
            )
        else:
            table_rows.append(f"{name} & {a:.1f} & {s_:.1f} & {rel:.1f} & {avg:.1f} \\\\")
    table_rows += [r"\bottomrule", r"\end{tabular}"]
    snippet = "\n".join(table_rows)
    (ROOT / "outputs/qwen35_9b_v6/dir_i_table.tex").write_text(snippet + "\n")

    # Replace Dir I table block in tex if placeholder numbers still present
    if "SELENE soft-share (v3)" in tex and claim_avg:
        new_block = r"""\subsection{Direction I --- Korean Affective Accuracy}
\textbf{Question:} Does Factor-LoRA Bank beat Blind/Scratch on full AI~Hub valid Avg?
Table~\ref{tab:dir_i} reports the v6 matrix (utterance S, $n{=}3182$).

\begin{table*}[!t]
\caption{Direction I: Factor-Bank matrix on AI~Hub valid ($n{=}3182$).}
\label{tab:dir_i}
\centering
""" + snippet + r"""
\end{table*}
"""
        tex = re.sub(
            r"\\subsection\{Direction I --- Korean Affective Accuracy\}.*?(?=\\subsection\{Direction II)",
            new_block,
            tex,
            count=1,
            flags=re.DOTALL,
        )

    if claim_avg:
        tex = tex.replace(
            "Thus absolute Dir~I Acc leadership is \\emph{not} claimed",
            f"Factor-Bank {winner} leads Dir~I Avg ({win_avg:.1f}\\% vs B16 {b16.get('Avg', 0):.1f}\\%)",
        )
    TEX.write_text(tex)
    print(f"updated {TEX} claim_avg={claim_avg}")


if __name__ == "__main__":
    main()
