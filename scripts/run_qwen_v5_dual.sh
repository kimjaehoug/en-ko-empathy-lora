#!/usr/bin/env bash
# SELENE v5: select_dual (EN-merge + KO LoRA) vs Blind/Scratch, full-valid eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MIN_FREE_GB="${MIN_FREE_GB:-18}"
PY="${PYTHON:-python3}"
OUT=outputs/qwen35_9b_v5
CFG=configs/qwen/stage3_ko_v5.yaml
mkdir -p "$OUT/stage2_gates"
cp -f outputs/qwen35_9b_v3/stage2_gates/gates.json "$OUT/stage2_gates/gates.json" 2>/dev/null \
  || cp -f outputs/qwen35_9b/stage2_gates/gates.json "$OUT/stage2_gates/gates.json"
GATES="$OUT/stage2_gates/gates.json"

"$PY" - <<PY
import os, time, torch
need=float(os.environ.get("MIN_FREE_GB","18"))
while True:
    free=torch.cuda.mem_get_info(0)[0]/1e9
    print(f"free_gb={free:.1f}", flush=True)
    if free>=need: break
    time.sleep(30)
print("GPU ready", torch.cuda.get_device_name(0))
PY

echo "=== SELECT v5 dual (EN merge + KO LoRA + gate losses) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode select_dual \
  --gates_file "$GATES" \
  --strategy_scope utterance \
  --en_replay_every 4 \
  --lora_anchor_weight 0.05 \
  --gate_conditioned_losses \
  --select_curriculum \
  --output_dir "$OUT/stage3_ko" \
  2>&1 | tee "$OUT/train_select.log"

echo "=== Scratch ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode relearn \
  --gates_file "$GATES" \
  --strategy_scope utterance \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses \
  --no_select_curriculum \
  --output_dir "$OUT/baseline_ko_scratch" \
  2>&1 | tee "$OUT/train_scratch.log"

echo "=== Blind share ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode share \
  --gates_file "$GATES" \
  --strategy_scope utterance \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses \
  --no_select_curriculum \
  --output_dir "$OUT/baseline_blind_share" \
  2>&1 | tee "$OUT/train_blind.log"

echo "=== Eval FULL ==="
"$PY" scripts/eval_direction_i_ii.py \
  --config configs/qwen/eval_direction_i_ii_v5.yaml --full \
  2>&1 | tee "$OUT/eval.log"

"$PY" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/qwen35_9b_v5/eval_direction_i_ii/report.json').read_text())
print('Dir I (full, utterance S)')
for k in ['select','ko_scratch','blind_share']:
    m=r['direction_i_ko'][k]['metrics']
    print(k, {x: round(100*m[x],2) for x in ['emotion_acc','strategy_acc','relation_acc']}, 'n=', m.get('n_examples'))
sel=r['direction_i_ko']['select']['metrics']
bli=r['direction_i_ko']['blind_share']['metrics']
print('Δ vs Blind', {k: round(100*(sel[k]-bli[k]),2) for k in ['emotion_acc','strategy_acc','relation_acc']})
b=r['direction_ii_en']['en_before_ko']['metrics']['emotion_acc']
for k in ['after_select','after_blind_share','after_ko_scratch']:
    a=r['direction_ii_en'][k]['metrics']['emotion_acc']
    print('DirII', k, round(100*a,1), 'Δ', round(100*(a-b),1))
PY
echo "DONE v5"
