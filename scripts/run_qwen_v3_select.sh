#!/usr/bin/env bash
# SELENE v3: make SELECT ≠ Blind via mixed gates + gate-conditioned losses
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MIN_FREE_GB="${MIN_FREE_GB:-20}"
PY="${PYTHON:-python3}"
OUT=outputs/qwen35_9b_v3
CFG=configs/qwen/stage3_ko_v3.yaml
mkdir -p "$OUT"

echo "=== rebuild mixed gates (A share, S/R relearn) ==="
"$PY" scripts/rebuild_gates_v3.py \
  --src outputs/qwen35_9b/stage2_gates/gates.json \
  --dst "$OUT/stage2_gates/gates.json"
GATES="$OUT/stage2_gates/gates.json"

"$PY" - <<PY
import os, time, torch
need=float(os.environ.get("MIN_FREE_GB","20"))
while True:
    free=torch.cuda.mem_get_info(0)[0]/1e9
    print(f"free_gb={free:.1f}", flush=True)
    if free>=need: break
    time.sleep(30)
print("GPU ready", torch.cuda.get_device_name(0))
PY

echo "=== SELECT (gate losses + curriculum + replay + soft_share) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode soft_share \
  --gates_file "$GATES" \
  --en_replay_every 4 \
  --lora_anchor_weight 0.08 \
  --gate_conditioned_losses \
  --select_curriculum \
  --output_dir "$OUT/stage3_ko" \
  2>&1 | tee "$OUT/train_select.log"

echo "=== Scratch (uniform losses, no SELECT extras) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode relearn \
  --gates_file "$GATES" \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses \
  --output_dir "$OUT/baseline_ko_scratch" \
  2>&1 | tee "$OUT/train_scratch.log"

echo "=== Blind share (uniform, no SELECT extras) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode share \
  --gates_file "$GATES" \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses \
  --output_dir "$OUT/baseline_blind_share" \
  2>&1 | tee "$OUT/train_blind.log"

echo "=== Eval ==="
"$PY" scripts/eval_direction_i_ii.py --config configs/qwen/eval_direction_i_ii_v3.yaml \
  2>&1 | tee "$OUT/eval.log"

echo "done -> $OUT/eval_direction_i_ii/report.json"
"$PY" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/qwen35_9b_v3/eval_direction_i_ii/report.json').read_text())
print('Dir I')
for k in ['select','ko_scratch','blind_share']:
    m=r['direction_i_ko'][k]['metrics']
    print(k, {x: round(100*m[x],2) for x in ['emotion_acc','strategy_acc','relation_acc']})
sel=r['direction_i_ko']['select']['metrics']
bli=r['direction_i_ko']['blind_share']['metrics']
scr=r['direction_i_ko']['ko_scratch']['metrics']
print('Δ vs Blind', {k: round(100*(sel[k]-bli[k]),2) for k in ['emotion_acc','strategy_acc','relation_acc']})
print('Δ vs Scratch', {k: round(100*(sel[k]-scr[k]),2) for k in ['emotion_acc','strategy_acc','relation_acc']})
b=r['direction_ii_en']['en_before_ko']['metrics']['emotion_acc']
for k in ['after_select','after_blind_share','after_ko_scratch']:
    a=r['direction_ii_en'][k]['metrics']['emotion_acc']
    print('DirII', k, round(100*a,1), 'Δ', round(100*(a-b),1))
PY
