#!/usr/bin/env bash
# TAFFC full experiment matrix: eval (KoED/confusion/error) + ablations + baselines + multi-seed + Llama
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MIN_FREE_GB="${MIN_FREE_GB:-18}"
PY="${PYTHON:-python3}"
LOG=outputs/taffc/run.log
mkdir -p outputs/taffc
exec > >(tee -a "$LOG") 2>&1

echo "=== TAFFC experiments $(date) ==="

has_ckpt() { test -f "$1/lora/adapter_model.safetensors"; }

run_train() {
  local cfg="$1" out="$2" seed="$3"
  shift 3
  if has_ckpt "$out"; then
    echo "[SKIP train] $out"
    return 0
  fi
  echo "[TRAIN] $out seed=$seed"
  "$PY" scripts/train_stage3_ko.py --config "$cfg" --output_dir "$out" --seed "$seed" "$@"
}

# ---- Phase 0: Extended eval on completed v6 + investigation ----
if [ "${SKIP_PHASE0:-0}" != "1" ]; then
echo "=== Phase 0: extended eval (v6) ==="
"$PY" scripts/eval_taffc_extended.py \
  --config configs/taffc/eval_taffc.yaml --full \
  --dump_predictions --error_analysis --koed \
  2>&1 | tee outputs/taffc/eval_v6_extended.log || true

"$PY" - <<'PY' || true
import json
from pathlib import Path
from src.eval.metrics import compare_pred_vectors
ROOT = Path('.')
pred = ROOT / 'outputs/taffc/eval/predictions'
if not pred.exists():
    print('no predictions yet'); raise SystemExit(0)
def load(name):
    p = pred / f'{name}_emotion_preds.json'
    if not p.exists(): return None
    d = json.loads(p.read_text())
    return d['pred'], d['gold']
for a,b in [('B16','B32'),('B16','S16'),('F16','B16')]:
    pa, ga = load(a) or (None, None)
    pb, gb = load(b) or (None, None)
    if pa and pb:
        c = compare_pred_vectors(pa, pb)
        print(f'{a} vs {b}: match={c["match_rate"]:.4f} identical={c["identical"]}')
        if ga and gb:
            acc_a = sum(x==y for x,y in zip(pa,ga))/len(pa)
            acc_b = sum(x==y for x,y in zip(pb,gb))/len(pb)
            print(f'  acc {a}={acc_a:.4f} {b}={acc_b:.4f}')
report = {'investigation': 'B16=B32=S16 A identical check'}
out = ROOT / 'outputs/taffc/investigation_b16.json'
if (pred / 'B16_emotion_preds.json').exists():
    pa,_ = load('B16'); pb,_ = load('B32'); ps,_ = load('S16')
    report['B16_vs_B32'] = compare_pred_vectors(pa, pb)
    report['B16_vs_S16'] = compare_pred_vectors(pa, ps)
    out.write_text(json.dumps(report, indent=2))
    print('wrote', out)
PY
fi

# ---- Phase 1: Baselines (MAD-X, KED translate-train) seed 42 ----
CFG=configs/qwen/stage3_ko_v6_bank.yaml
GATES=outputs/qwen35_9b_v3/stage2_gates/gates.json

echo "=== Phase 1: MAD-X + KED baselines ==="
run_train "$CFG" outputs/taffc/qwen/MADX/s42 42 \
  --gates_file "$GATES" --init_mode madx --lora_r 16 \
  --strategy_scope utterance --no_gate_conditioned_losses --no_select_curriculum \
  --en_replay_every 0 --lora_anchor_weight 0.0

run_train configs/taffc/stage3_ked.yaml outputs/taffc/qwen/KED/s42 42 \
  --gates_file "$GATES" --init_mode relearn --lora_r 16 \
  --strategy_scope utterance --no_gate_conditioned_losses --no_select_curriculum \
  --en_replay_every 0 --lora_anchor_weight 0.0

# ---- Phase 2: Ablation matrix (seed 42) ----
echo "=== Phase 2: ablations ==="
ABL=(
  "A0_v3_soft:outputs/taffc/qwen/ablation/A0_v3_soft/s42:--init_mode soft_share --lora_r 16 --gate_conditioned_losses --select_curriculum --en_replay_every 4 --lora_anchor_weight 0.04"
  "A2_no_two_pass:outputs/taffc/qwen/ablation/A2_no_two_pass/s42:--init_mode select_bank --lora_r 16 --no_two_pass_affect --gate_conditioned_losses --select_curriculum --en_replay_every 4 --lora_anchor_weight 0.04"
  "A3_no_gate:outputs/taffc/qwen/ablation/A3_no_gate/s42:--init_mode select_bank --lora_r 16 --two_pass_affect --no_gate_conditioned_losses --no_select_curriculum --en_replay_every 4 --lora_anchor_weight 0.04"
  "A4_no_curr:outputs/taffc/qwen/ablation/A4_no_curr/s42:--init_mode select_bank --lora_r 16 --two_pass_affect --gate_conditioned_losses --no_select_curriculum --en_replay_every 4 --lora_anchor_weight 0.04"
  "A5_no_replay:outputs/taffc/qwen/ablation/A5_no_replay/s42:--init_mode select_bank --lora_r 16 --two_pass_affect --gate_conditioned_losses --select_curriculum --en_replay_every 0 --lora_anchor_weight 0.04"
  "A6_no_anchor:outputs/taffc/qwen/ablation/A6_no_anchor/s42:--init_mode select_bank --lora_r 16 --two_pass_affect --gate_conditioned_losses --select_curriculum --en_replay_every 4 --lora_anchor_weight 0.0"
  "A8_freeze:outputs/taffc/qwen/ablation/A8_freeze/s42:--init_mode select --lora_r 16 --freeze_lora --gate_conditioned_losses --select_curriculum --en_replay_every 4"
  "A9_lm_only:outputs/taffc/qwen/ablation/A9_lm_only/s42:--init_mode select_bank --lora_r 16 --two_pass_affect --lm_only"
)
for spec in "${ABL[@]}"; do
  IFS=: read -r name out _ <<< "$spec"
  extra="${spec#*:*:}"
  # shellcheck disable=SC2086
  run_train "$CFG" "$out" 42 --gates_file "$GATES" --strategy_scope utterance $extra
done

# ---- Phase 3: Multi-seed main systems (seeds 123, 456; 42 from v6/taffc) ----
echo "=== Phase 3: multi-seed F16/B16/S16 ==="
for seed in 123 456; do
  for spec in \
    "F16:--init_mode select_bank --lora_r 16 --two_pass_affect --gate_conditioned_losses --select_curriculum --en_replay_every 4 --lora_anchor_weight 0.04" \
    "B16:--init_mode share --lora_r 16 --no_gate_conditioned_losses --no_select_curriculum --en_replay_every 0 --lora_anchor_weight 0.0" \
    "S16:--init_mode relearn --lora_r 16 --no_gate_conditioned_losses --no_select_curriculum --en_replay_every 0 --lora_anchor_weight 0.0"; do
    IFS=: read -r rid extra <<< "$spec"
    out="outputs/taffc/qwen/${rid}/s${seed}"
    # shellcheck disable=SC2086
    run_train "$CFG" "$out" "$seed" --gates_file "$GATES" --strategy_scope utterance $extra
  done
done

# ---- Phase 4: Llama backbone (generalization) ----
echo "=== Phase 4: Llama-3.1-8B ==="
if [ -d outputs/llama31_8b/stage1_en/lora ]; then
  for seed in 42 123 456; do
    run_train configs/llama/stage3_ko_taffc.yaml "outputs/taffc/llama/F16/s${seed}" "$seed" \
      --init_mode select_bank --lora_r 16 --two_pass_affect \
      --gate_conditioned_losses --select_curriculum --en_replay_every 4 --lora_anchor_weight 0.04
    run_train configs/llama/stage3_ko_taffc.yaml "outputs/taffc/llama/B16/s${seed}" "$seed" \
      --init_mode share --lora_r 16 --no_gate_conditioned_losses --no_select_curriculum \
      --en_replay_every 0 --lora_anchor_weight 0.0
  done
else
  echo "[SKIP Llama] stage1 not found — run stage1_en first"
fi

# ---- Phase 5: Final extended eval + aggregate ----
echo "=== Phase 5: final eval + aggregate ==="
"$PY" scripts/eval_taffc_extended.py --config configs/taffc/eval_taffc.yaml --full \
  --dump_predictions --error_analysis --koed 2>&1 | tee outputs/taffc/eval_final.log

"$PY" scripts/aggregate_taffc_results.py
"$PY" scripts/fill_taffc_paper_results.py 2>/dev/null || "$PY" scripts/fill_v6_paper_results.py 2>/dev/null || true

echo "DONE TAFFC $(date)"
