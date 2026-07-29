#!/usr/bin/env bash
# Train KO baselines for Direction I/II, then evaluate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/Library/TinyTeX/bin/universal-darwin:${PATH:-}"

STEPS="${1:-50}"

echo "== KO-scratch (untouched KO / no EN init) =="
python3 scripts/train_stage3_ko.py \
  --init_mode relearn \
  --output_dir outputs/baseline_ko_scratch \
  --max_steps "$STEPS"

echo "== Blind share (EN LoRA copy, no factor gate policy) =="
python3 scripts/train_stage3_ko.py \
  --init_mode share \
  --output_dir outputs/baseline_blind_share \
  --max_steps "$STEPS"

echo "== Evaluate Direction I/II =="
python3 scripts/eval_direction_i_ii.py --max_batches 50

echo "done -> outputs/eval_direction_i_ii/report.json"
