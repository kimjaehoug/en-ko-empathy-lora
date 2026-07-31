#!/usr/bin/env bash
# SELENE v4: full-valid protocol + factor-modular SELECT (freeze EN LoRA)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MIN_FREE_GB="${MIN_FREE_GB:-20}"
PY="${PYTHON:-python3}"
OUT=outputs/qwen35_9b_v4
CFG=configs/qwen/stage3_ko_v4.yaml
mkdir -p "$OUT"

GATES_SRC=outputs/qwen35_9b_v3/stage2_gates/gates.json
if [[ ! -f "$GATES_SRC" ]]; then
  GATES_SRC=outputs/qwen35_9b/stage2_gates/gates.json
fi
mkdir -p "$OUT/stage2_gates"
cp -f "$GATES_SRC" "$OUT/stage2_gates/gates.json"
GATES="$OUT/stage2_gates/gates.json"
echo "gates=$(python3 -c "import json;g=json.load(open('$GATES'))['gates'];print({k:v['decision'] for k,v in g.items()})")"

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

echo "=== [0] Re-eval v3 on FULL valid (protocol check) ==="
"$PY" scripts/eval_direction_i_ii.py \
  --config configs/qwen/eval_direction_i_ii_v3_full.yaml --full \
  2>&1 | tee "$OUT/eval_v3_full.log"
"$PY" - <<'PY'
import json
from pathlib import Path
p=Path('outputs/qwen35_9b_v3/eval_direction_i_ii_full/report.json')
if p.exists():
    r=json.loads(p.read_text())
    print('V3 FULL Dir I')
    for k in ['select','ko_scratch','blind_share']:
        m=r['direction_i_ko'][k]['metrics']
        print(k, {x: round(100*m[x],2) for x in ['emotion_acc','strategy_acc','relation_acc']},
              'n=', m.get('n_examples'))
    b=r['direction_ii_en']['en_before_ko']['metrics']['emotion_acc']
    for k in ['after_select','after_blind_share','after_ko_scratch']:
        a=r['direction_ii_en'][k]['metrics']['emotion_acc']
        print('DirII', k, round(100*a,1), 'Δ', round(100*(a-b),1),
              'n=', r['direction_ii_en'][k]['metrics'].get('n_examples'))
PY

echo "=== SELECT v4 (freeze EN LoRA + gate head LR/loss + session S) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode select \
  --gates_file "$GATES" \
  --freeze_lora \
  --strategy_scope session \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --gate_conditioned_losses \
  --select_curriculum \
  --output_dir "$OUT/stage3_ko" \
  2>&1 | tee "$OUT/train_select.log"

echo "=== Scratch (fresh LoRA, uniform, no SELECT extras) ==="
"$PY" scripts/train_stage3_ko.py \
  --config "$CFG" \
  --init_mode relearn \
  --gates_file "$GATES" \
  --strategy_scope session \
  --en_replay_every 0 \
  --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses \
  --no_select_curriculum \
  --output_dir "$OUT/baseline_ko_scratch" \
  2>&1 | tee "$OUT/train_scratch.log"

echo "=== Blind share (train EN LoRA, uniform, no SELECT extras) ==="
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

echo "=== Eval FULL valid ==="
"$PY" scripts/eval_direction_i_ii.py \
  --config configs/qwen/eval_direction_i_ii_v4.yaml --full \
  2>&1 | tee "$OUT/eval.log"

echo "done -> $OUT/eval_direction_i_ii/report.json"
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
