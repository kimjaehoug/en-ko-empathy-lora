#!/usr/bin/env bash
# 실시간 학습 진행 상황 보기
# Usage: bash scripts/watch_progress.sh [stage_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:-$ROOT/outputs/qwen35_9b/stage3_ko}"
PROGRESS="$DIR/progress.json"
if [[ ! -f "$PROGRESS" ]]; then
  echo "progress.json 없음: $PROGRESS"
  echo "예: bash scripts/watch_progress.sh outputs/qwen35_9b/stage1_en"
  exit 1
fi
echo "watching $PROGRESS (Ctrl+C 종료)"
watch -n 5 "python3 - <<'PY'
import json
from pathlib import Path
p = Path('$PROGRESS')
d = json.loads(p.read_text())
print('stage:', d.get('stage'), '| status:', d.get('status'))
print('step:', d.get('global_step'), '/', d.get('total_steps'), '| elapsed:', d.get('elapsed_sec'), 's')
if 'train' in d:
    t = d['train']
    acc = {k:v for k,v in t.items() if k.endswith('_acc')}
    loss = {k:v for k,v in t.items() if 'loss' in k}
    print('train loss:', loss)
    print('train acc:', acc)
for k in sorted(d):
    if k.startswith('eval_'):
        print(k, d[k])
if 'final_eval' in d:
    print('FINAL', d['final_eval'])
PY"
