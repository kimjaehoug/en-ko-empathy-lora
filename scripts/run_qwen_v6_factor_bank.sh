#!/usr/bin/env bash
# Factor-LoRA Bank Dir-I matrix: F16, S16, B16, F32, B32 + full-valid eval
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MIN_FREE_GB="${MIN_FREE_GB:-18}"
PY="${PYTHON:-python3}"
OUT=outputs/qwen35_9b_v6
CFG=configs/qwen/stage3_ko_v6_bank.yaml
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

run_train() {
  local name="$1"; shift
  echo "=== $name ==="
  "$PY" scripts/train_stage3_ko.py --config "$CFG" --gates_file "$GATES" "$@" \
    --output_dir "$OUT/$name" 2>&1 | tee "$OUT/train_${name}.log"
}

# F16: Factor bank, fair capacity, two-pass A
run_train F16 \
  --init_mode select_bank --lora_r 16 --lora_alpha 32 \
  --strategy_scope utterance --two_pass_affect \
  --en_replay_every 4 --lora_anchor_weight 0.04 \
  --gate_conditioned_losses --select_curriculum

# S16: Scratch fair
run_train S16 \
  --init_mode relearn --lora_r 16 --lora_alpha 32 \
  --strategy_scope utterance \
  --en_replay_every 0 --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses --no_select_curriculum

# B16: Blind fair
run_train B16 \
  --init_mode share --lora_r 16 --lora_alpha 32 \
  --strategy_scope utterance \
  --en_replay_every 0 --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses --no_select_curriculum

# F32: Factor bank capacity↑ (+MLP targets)
run_train F32 \
  --init_mode select_bank --lora_r 32 --lora_alpha 64 --lora_include_mlp \
  --strategy_scope utterance --two_pass_affect \
  --en_replay_every 4 --lora_anchor_weight 0.04 \
  --gate_conditioned_losses --select_curriculum

# B32: Blind matched capacity control
run_train B32 \
  --init_mode share --lora_r 32 --lora_alpha 64 --lora_include_mlp \
  --strategy_scope utterance \
  --en_replay_every 0 --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses --no_select_curriculum

echo "=== Eval FULL matrix ==="
"$PY" scripts/eval_direction_i_ii.py \
  --config configs/qwen/eval_direction_i_ii_v6.yaml --full \
  2>&1 | tee "$OUT/eval.log"

"$PY" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/qwen35_9b_v6/eval_direction_i_ii/report.json').read_text())
print('Dir I (full, utterance)')
rows=[]
for k in ['F16','F32','B16','B32','S16']:
    blob=r['direction_i_ko'].get(k) or {}
    m=blob.get('metrics') or {}
    if not m: 
        print(k, 'MISSING'); continue
    a,s,rel=100*m['emotion_acc'],100*m['strategy_acc'],100*m['relation_acc']
    avg=(a+s+rel)/3
    rows.append((k,a,s,rel,avg,m.get('n_examples')))
    print(f"{k}: A={a:.2f} S={s:.2f} R={rel:.2f} Avg={avg:.2f} n={m.get('n_examples')}")
rows_sorted=sorted(rows, key=lambda x: -x[4])
print('Avg rank:', [x[0] for x in rows_sorted])
# per-axis winners
for name,idx in [('A',1),('S',2),('R',3)]:
    best=max(rows, key=lambda x: x[idx])
    print(f'best_{name}', best[0], round(best[idx],2))
b=r['direction_ii_en']['en_before_ko']['metrics']['emotion_acc']
print('Dir II (before', round(100*b,1), ')')
for k in ['after_F16','after_F32','after_B16','after_B32','after_S16']:
    m=(r['direction_ii_en'].get(k) or {}).get('metrics') or {}
    if not m: continue
    a=m['emotion_acc']
    print(k, round(100*a,1), 'Δ', round(100*(a-b),1))
Path('outputs/qwen35_9b_v6/summary.json').write_text(json.dumps({
    'dir_i':[{'id':k,'A':a,'S':s,'R':rel,'Avg':avg,'n':n} for k,a,s,rel,avg,n in rows],
    'avg_rank':[x[0] for x in rows_sorted],
}, indent=2))
print('wrote outputs/qwen35_9b_v6/summary.json')
PY
echo "DONE v6"
