#!/usr/bin/env bash
# SELENE Qwen3.5-9B full pipeline on NVIDIA L40S (bf16 LoRA, no QLoRA).
#
# Usage:
#   bash scripts/run_qwen_full.sh              # full epochs
#   bash scripts/run_qwen_full.sh smoke        # Stage1 20-step smoke only
#   bash scripts/run_qwen_full.sh full 200     # cap Stage1/3 max_steps at 200
#
# Env:
#   CUDA_VISIBLE_DEVICES   default 0 (free L40S)
#   PYTHON                 default python3
#   HF_TOKEN / HUGGING_FACE_HUB_TOKEN  if Hub download needs auth
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python3}"
MODE="${1:-full}"
STEPS_CAP="${2:-}"

CFG1="configs/qwen/stage1_en.yaml"
CFG2="configs/qwen/stage2_gates.yaml"
CFG3="configs/qwen/stage3_ko.yaml"
CFGE="configs/qwen/eval_direction_i_ii.yaml"
OUT="outputs/qwen35_9b"
mkdir -p "$OUT"

echo "=== env ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required for Qwen3.5-9B LoRA"
name = torch.cuda.get_device_name(0)
print("cuda", True, name, round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), "GB")
if "L40S" not in name and "L40" not in name:
    print("WARNING: expected NVIDIA L40S, got", name)
PY

step_args=()
if [[ -n "$STEPS_CAP" ]]; then
  step_args=(--max_steps "$STEPS_CAP")
fi

if [[ "$MODE" == "smoke" ]]; then
  echo "=== Phase A smoke: Stage1 max_steps=20 ==="
  "$PY" scripts/train_stage1_en.py --config "$CFG1" --max_steps 20
  echo "smoke ok -> $OUT/stage1_en"
  exit 0
fi

echo "=== Stage1 EN ==="
"$PY" scripts/train_stage1_en.py --config "$CFG1" "${step_args[@]}"

echo "=== Stage2 gates ==="
"$PY" scripts/run_stage2_gates.py --config "$CFG2"
"$PY" - <<'PY'
import json, sys
from pathlib import Path
p = Path("outputs/qwen35_9b/stage2_gates/gates.json")
blob = json.loads(p.read_text())
dec = {k: v["decision"] for k, v in blob["gates"].items()}
print("gates", dec)
if all(v == "share" for v in dec.values()):
    print("FATAL: all-share gate collapse", file=sys.stderr)
    sys.exit(2)
if dec.get("strategy") == "share" and dec.get("relation") == "share":
    print("WARNING: S and R both share; SELECT may ≈ Blind share")
PY

echo "=== Stage3 SELECT (auto + gates) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG3" \
  --init_mode auto \
  --gates_file "$OUT/stage2_gates/gates.json" \
  --output_dir "$OUT/stage3_ko" \
  "${step_args[@]}"

echo "=== Baseline KO-scratch (relearn) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG3" \
  --init_mode relearn \
  --gates_file "$OUT/stage2_gates/gates.json" \
  --output_dir "$OUT/baseline_ko_scratch" \
  "${step_args[@]}"

echo "=== Baseline Blind share ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG3" \
  --init_mode share \
  --gates_file "$OUT/stage2_gates/gates.json" \
  --output_dir "$OUT/baseline_blind_share" \
  "${step_args[@]}"

echo "=== Eval Direction I/II ==="
"$PY" scripts/eval_direction_i_ii.py --config "$CFGE"

echo "done."
echo "  gates:  $OUT/stage2_gates/gates.json"
echo "  report: $OUT/eval_direction_i_ii/report.json"
