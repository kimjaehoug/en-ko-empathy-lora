#!/usr/bin/env bash
# Resume: Phase 0 eval (fixed) then Phase 1-5 with SKIP_PHASE0=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python3}"

echo "=== TAFFC RESUME $(date) ==="
echo "=== Phase 0: extended eval (retry after bugfix) ==="
"$PY" scripts/eval_taffc_extended.py \
  --config configs/taffc/eval_taffc.yaml --full \
  --dump_predictions --error_analysis --koed \
  2>&1 | tee -a outputs/taffc/eval_v6_extended.log

"$PY" - <<'PY' || true
import json
from pathlib import Path
from src.eval.metrics import compare_pred_vectors
ROOT = Path('.')
def load(name):
    p = ROOT / f'outputs/taffc/eval/predictions/{name}_emotion_preds.json'
    if not p.exists(): return None, None
    d = json.loads(p.read_text())
    return d['pred'], d['gold']
pa, _ = load('B16'); pb, _ = load('B32'); ps, _ = load('S16')
report = {}
if pa and pb: report['B16_vs_B32'] = compare_pred_vectors(pa, pb)
if pa and ps: report['B16_vs_S16'] = compare_pred_vectors(pa, ps)
Path('outputs/taffc/investigation_b16.json').write_text(json.dumps(report, indent=2))
print('investigation', report)
PY

export SKIP_PHASE0=1
exec bash scripts/run_taffc_experiments.sh
