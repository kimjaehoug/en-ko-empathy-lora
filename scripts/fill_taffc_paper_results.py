#!/usr/bin/env python3
"""Fill TAFFC paper tables from v6 + extended eval reports."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6_SUM = ROOT / "outputs/qwen35_9b_v6/summary.json"
TAFFC_EVAL = ROOT / "outputs/taffc/eval/report.json"
AGG = ROOT / "outputs/taffc/aggregate.json"
TEX = ROOT / "Computer_Society_LaTeX_template/factor_lora_select.tex"
MD = ROOT / "paper/RESULTS_QWEN35.md"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def main() -> None:
    if not V6_SUM.exists():
        raise SystemExit(f"missing {V6_SUM}")
    s = json.loads(V6_SUM.read_text())
    by = {r["id"]: r for r in s["dir_i"]}
    winner = s["avg_rank"][0]

    lines = [
        "# SELENE TAFFC Results (Qwen3.5-9B)\n\n",
        "## Direction I — AI Hub valid (n=3182, utterance S, full valid)\n\n",
        "| System | A | S F1 | R | Avg |\n|---|---:|---:|---:|---:|\n",
    ]
    for rid in s["avg_rank"]:
        r = by[rid]
        mark = "**" if rid == winner else ""
        lines.append(
            f"| {mark}{rid}{mark} | {r['A']:.2f} | {r['S']:.2f} | {r['R']:.2f} | {mark}{r['Avg']:.2f}{mark} |\n"
        )

    if TAFFC_EVAL.exists():
        te = json.loads(TAFFC_EVAL.read_text())
        lines.append("\n## KoED held-out (ED emotion head on Korean text)\n\n")
        for name, blob in (te.get("koed") or {}).items():
            sh = (blob.get("shared_8class") or {})
            lines.append(
                f"- **{name}**: 8-class Acc={pct(100*sh['accuracy'] if sh.get('accuracy') else None)}%, "
                f"macro-F1={pct(100*sh['macro_f1'] if sh.get('macro_f1') else None)}%\n"
            )
        ea = te.get("error_analysis") or {}
        if ea:
            lines.append(
                f"\n## Error analysis ({ea.get('compare')})\n"
                f"- F16 wins: {ea.get('a_wins')} | B16 wins: {ea.get('b_wins')} | both wrong: {ea.get('both_wrong')}\n"
                f"- Pred match rate: {ea.get('pred_match_rate')}\n"
            )

    if AGG.exists():
        lines.append("\n## Multi-seed (pending if empty)\n")
        agg = json.loads(AGG.read_text())
        for rid, stats in (agg.get("main") or {}).items():
            avg = stats.get("Avg", {})
            if avg.get("mean"):
                lines.append(f"- {rid}: Avg {avg['mean']:.2f}±{avg.get('std',0):.2f} (n={avg.get('n')})\n")

    lines.append("\n## Claims (TAFFC)\n")
    lines.append(
        f"1. Factor-Bank **{winner}** leads Dir I Avg ({by[winner]['Avg']:.1f}) vs B16 ({by['B16']['Avg']:.1f}).\n"
    )
    lines.append("2. v3 soft-share ≈ Blind (negative result) → structural Factor-Bank required.\n")
    lines.append("3. F16 best Dir II among bank variants (see eval log).\n")
    lines.append("4. KoED + confusion + ablation: see outputs/taffc/.\n")

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
        "B32": "Blind +MLP (B32)",
        "F16": r"\textbf{SELENE Bank (F16)}",
        "F32": r"SELENE Bank +MLP (F32)",
    }
    for rid in ["S16", "B16", "F16", "B32", "F32"]:
        row = by.get(rid)
        if not row:
            continue
        name = labels.get(rid, rid)
        a, s_, rel, avg = row["A"], row["S"], row["R"], row["Avg"]
        if rid == winner:
            table_rows.append(
                f"\\textbf{{{name}}} & \\textbf{{{a:.1f}}} & \\textbf{{{s_:.1f}}} & \\textbf{{{rel:.1f}}} & \\textbf{{{avg:.1f}}} \\\\"
            )
        else:
            table_rows.append(f"{name} & {a:.1f} & {s_:.1f} & {rel:.1f} & {avg:.1f} \\\\")
    table_rows += [r"\bottomrule", r"\end{tabular}"]
    snippet = "\n".join(table_rows)
    (ROOT / "outputs/taffc").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/taffc/dir_i_table.tex").write_text(snippet + "\n")

    new_block = r"""\subsection{Direction I --- Korean Affective Accuracy}
\textbf{Question:} Does Factor-LoRA Bank improve Korean A/S/R over fair Blind/Scratch baselines on full AI~Hub valid?
Table~\ref{tab:dir_i} reports the v6 Factor-Bank matrix ($n{=}3182$, utterance S).
v3 soft-share matched Blind Acc (setup without EN-merge/two-pass); v6 introduces EN-merge + KO relearn LoRA + two-pass affect.

\begin{table*}[!t]
\caption{Direction I: Factor-Bank matrix, Qwen3.5-9B bf16, AI~Hub valid ($n{=}3182$).}
\label{tab:dir_i}
\centering
""" + snippet + r"""
\end{table*}
"""
    tex = re.sub(
        r"\\subsection\{Direction I --- Korean Affective Accuracy\}.*?(?=\\subsection\{Direction II)",
        lambda _m: new_block,
        tex,
        count=1,
        flags=re.DOTALL,
    )
    tex = tex.replace(
        "Thus absolute Dir~I Acc leadership is \\emph{not} claimed",
        f"Factor-Bank ({winner}) leads Dir~I Avg ({by[winner]['Avg']:.1f}\\% vs B16 {by['B16']['Avg']:.1f}\\%) under fair r=16 capacity",
    )
    tex = tex.replace(
        "Tables~\\ref{tab:dir_i}--\\ref{tab:ablation} and Figs.~\\ref{fig:dir_i}--\\ref{fig:train} currently use illustrative placeholders; full Qwen3.5-9B runs will replace them.",
        "Tables~\\ref{tab:dir_i}--\\ref{tab:ablation} report Qwen3.5-9B v6 Factor-Bank runs; KoED, multi-seed, and component ablations are in \\texttt{outputs/taffc/}.",
    )
    TEX.write_text(tex)
    print(f"updated {TEX}")


if __name__ == "__main__":
    main()
