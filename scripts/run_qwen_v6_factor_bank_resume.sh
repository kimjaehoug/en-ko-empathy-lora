#!/usr/bin/env bash
# Resume v6 matrix from first incomplete run (F32/B32/eval).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MIN_FREE_GB="${MIN_FREE_GB:-18}"
PY="${PYTHON:-python3}"
OUT=outputs/qwen35_9b_v6
CFG=configs/qwen/stage3_ko_v6_bank.yaml
GATES="$OUT/stage2_gates/gates.json"
mkdir -p "$OUT/stage2_gates"
cp -f outputs/qwen35_9b_v3/stage2_gates/gates.json "$GATES" 2>/dev/null || true

has_ckpt() {
  local name="$1"
  test -f "$OUT/$name/lora/adapter_model.safetensors" \
    || test -f "$OUT/$name/checkpoint-final/lora/adapter_model.safetensors"
}

run_train() {
  local name="$1"; shift
  if has_ckpt "$name"; then
    echo "=== $name SKIP (checkpoint exists) ==="
    return 0
  fi
  echo "=== $name ==="
  rm -rf "$OUT/$name"
  "$PY" scripts/train_stage3_ko.py --config "$CFG" --gates_file "$GATES" "$@" \
    --output_dir "$OUT/$name" 2>&1 | tee -a "$OUT/train_${name}.log"
}

run_train F32 \
  --init_mode select_bank --lora_r 32 --lora_alpha 64 --lora_include_mlp \
  --strategy_scope utterance --two_pass_affect \
  --en_replay_every 4 --lora_anchor_weight 0.04 \
  --gate_conditioned_losses --select_curriculum

run_train B32 \
  --init_mode share --lora_r 32 --lora_alpha 64 --lora_include_mlp \
  --strategy_scope utterance \
  --en_replay_every 0 --lora_anchor_weight 0.0 \
  --no_gate_conditioned_losses --no_select_curriculum

if [ ! -f "$OUT/eval_direction_i_ii/report.json" ]; then
  echo "=== Eval FULL matrix ==="
  "$PY" scripts/eval_direction_i_ii.py \
    --config configs/qwen/eval_direction_i_ii_v6.yaml --full \
    2>&1 | tee -a "$OUT/eval.log"
fi

"$PY" scripts/fill_v6_paper_results.py 2>/dev/null || "$PY" - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('outputs/qwen35_9b_v6/eval_direction_i_ii/report.json').read_text())
rows=[]
for k in ['F16','F32','B16','B32','S16']:
    m=(r['direction_i_ko'].get(k) or {}).get('metrics') or {}
    if not m: continue
    a,s,rel=100*m['emotion_acc'],100*m['strategy_acc'],100*m['relation_acc']
    rows.append((k,a,s,rel,(a+s+rel)/3,m.get('n_examples')))
    print(k, f"A={a:.2f} S={s:.2f} R={rel:.2f} Avg={(a+s+rel)/3:.2f}")
Path('outputs/qwen35_9b_v6/summary.json').write_text(json.dumps({
    'dir_i':[{'id':k,'A':a,'S':s,'R':rel,'Avg':avg,'n':n} for k,a,s,rel,avg,n in rows],
    'avg_rank':[x[0] for x in sorted(rows, key=lambda x:-x[4])],
}, indent=2))
PY
echo "DONE v6 resume"
