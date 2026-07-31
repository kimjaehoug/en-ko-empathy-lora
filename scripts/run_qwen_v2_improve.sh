#!/usr/bin/env bash
# SELENE v2: multi-label S + soft-share + EN replay + LoRA anchor
# Reuses Stage1/Stage2 from outputs/qwen35_9b/; writes outputs/qwen35_9b_v2/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python3}"
OUT=outputs/qwen35_9b_v2
CFG=configs/qwen/stage3_ko_v2.yaml
GATES=outputs/qwen35_9b/stage2_gates/gates.json
MIN_FREE_GB="${MIN_FREE_GB:-32}"
mkdir -p "$OUT"

echo "=== waiting for >=${MIN_FREE_GB}GB free on visible GPU ==="
"$PY" - <<PY
import os, time, torch
need = float(os.environ.get("MIN_FREE_GB", "32"))
assert torch.cuda.is_available()
while True:
    free = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"free_gb={free:.1f} need={need}", flush=True)
    if free >= need:
        break
    time.sleep(60)
print("GPU ready:", torch.cuda.get_device_name(0))
PY

echo "=== Stage3 SELECT soft_share + EN replay + anchor (v2) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode soft_share \
  --gates_file "$GATES" \
  --en_replay_every 4 \
  --lora_anchor_weight 0.05 \
  --output_dir "$OUT/stage3_ko" \
  2>&1 | tee "$OUT/train_select.log"

echo "=== Baseline KO-scratch (relearn, multilabel S, no EN replay) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode relearn \
  --gates_file "$GATES" \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --output_dir "$OUT/baseline_ko_scratch" \
  2>&1 | tee "$OUT/train_scratch.log"

echo "=== Baseline Blind share (EN copy, no replay/anchor) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode share \
  --gates_file "$GATES" \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --output_dir "$OUT/baseline_blind_share" \
  2>&1 | tee "$OUT/train_blind.log"

echo "=== Eval Dir I/II v2 ==="
"$PY" scripts/eval_direction_i_ii.py --config configs/qwen/eval_direction_i_ii_v2.yaml \
  2>&1 | tee "$OUT/eval.log"

echo "done -> $OUT/eval_direction_i_ii/report.json"
