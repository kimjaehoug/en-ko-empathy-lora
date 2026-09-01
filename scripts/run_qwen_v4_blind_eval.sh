#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MIN_FREE_GB="${MIN_FREE_GB:-18}"
PY="${PYTHON:-python3}"
OUT=outputs/qwen35_9b_v4
CFG=configs/qwen/stage3_ko_v4.yaml
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

echo "=== Blind share (resume/complete) ==="
rm -rf "$OUT/baseline_blind_share"
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode share \
  --gates_file "$GATES" \
  --strategy_scope session \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses \
  --no_select_curriculum \
  --output_dir "$OUT/baseline_blind_share" \
  2>&1 | tee "$OUT/train_blind.log"

echo "=== Eval FULL valid v4 ==="
"$PY" scripts/eval_direction_i_ii.py \
  --config configs/qwen/eval_direction_i_ii_v4.yaml --full \
  2>&1 | tee "$OUT/eval.log"

"$PY" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/qwen35_9b_v4/eval_direction_i_ii/report.json').read_text())
print('Dir I (full)')
for k in ['select','ko_scratch','blind_share']:
    m=r['direction_i_ko'][k]['metrics']
    print(k, {x: round(100*m[x],2) for x in ['emotion_acc','strategy_acc','relation_acc']},
          'exactS', round(100*m.get('strategy_exact_match',0),2),
          'n=', m.get('n_examples'))
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
echo "DONE v4"
