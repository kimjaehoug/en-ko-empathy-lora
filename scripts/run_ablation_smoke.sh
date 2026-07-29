#!/usr/bin/env bash
# GPT-2 smoke: Stage-wise ablations A0–A8 (50 steps unless noted).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
STEPS="${MAX_STEPS:-50}"
EVAL_BATCHES="${MAX_BATCHES:-30}"

mkdir -p outputs/ablation

echo "== Prepare A0/A1/A2 from existing runs (symlink) =="
link_ckpt() {
  local src="$1" dst="$2"
  rm -rf "$dst"
  mkdir -p "$dst"
  for name in lora heads.pt labels.json tokenizer run_meta.json train_history.json; do
    if [[ -e "$src/$name" ]]; then
      ln -sfn "$(cd "$src" && pwd)/$name" "$dst/$name"
    fi
  done
}
link_ckpt outputs/stage3_ko outputs/ablation/A0_full_select
link_ckpt outputs/baseline_ko_scratch outputs/ablation/A1_wo_stage1
link_ckpt outputs/baseline_blind_share outputs/ablation/A2_wo_stage2

echo "== A7 gates: A from Stage2; S/R forced relearn =="
"$PY" - <<'PY'
import json
from pathlib import Path
src = Path("outputs/stage2_gates/gates.json")
blob = json.loads(src.read_text())
gates = blob["gates"]
for k in ("strategy", "relation"):
    gates[k] = {**gates[k], "decision": "relearn", "forced": True}
out = Path("outputs/ablation/A7_gates.json")
out.parent.mkdir(parents=True, exist_ok=True)
blob["note"] = "A7: affect keeps Stage2 decision; S/R forced relearn; init=affect_priority"
out.write_text(json.dumps(blob, ensure_ascii=False, indent=2))
print("wrote", out)
PY

echo "== A3 w/o Stage3 heads (LM-only) =="
"$PY" scripts/train_stage3_ko.py \
  --init_mode auto \
  --lm_only \
  --max_steps "$STEPS" \
  --output_dir outputs/ablation/A3_wo_stage3_heads

echo "== A4 Stage1 only (EN LoRA, 0 KO steps) =="
"$PY" scripts/train_stage3_ko.py \
  --init_mode share \
  --max_steps 0 \
  --output_dir outputs/ablation/A4_stage1_only

echo "== A5 all-relearn =="
"$PY" scripts/train_stage3_ko.py \
  --init_mode relearn \
  --max_steps "$STEPS" \
  --output_dir outputs/ablation/A5_all_relearn

echo "== A6 all-suppress =="
"$PY" scripts/train_stage3_ko.py \
  --init_mode suppress \
  --max_steps "$STEPS" \
  --output_dir outputs/ablation/A6_all_suppress

echo "== A7 gate A only (affect_priority) =="
"$PY" scripts/train_stage3_ko.py \
  --init_mode affect_priority \
  --gates_file outputs/ablation/A7_gates.json \
  --max_steps "$STEPS" \
  --output_dir outputs/ablation/A7_gate_A_only

echo "== A8 compose α: reuse A0 LoRA; Dir II α stub recorded in eval =="
link_ckpt outputs/stage3_ko outputs/ablation/A8_compose_alpha

echo "== Evaluate ablation Dir I/II =="
"$PY" scripts/eval_direction_i_ii.py \
  --config configs/eval_ablation_stages.yaml \
  --max_batches "$EVAL_BATCHES"

echo "Done. Report: outputs/eval_ablation_stages/report.json"
