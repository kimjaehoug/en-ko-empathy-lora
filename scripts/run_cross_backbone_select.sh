#!/usr/bin/env bash
# Cross-backbone Factor-LoRA SELECT (same recipe on Qwen / Llama / EXAONE).
#
# Goal: measure whether SELECT−Blind / SELECT−Scratch ΔAcc depends on backbone.
# Protocol matches configs/{family}/ (Stage1→2→3 SELECT + scratch + blind + eval).
#
# Usage:
#   bash scripts/run_cross_backbone_select.sh qwen            # already have outputs/qwen35_9b
#   bash scripts/run_cross_backbone_select.sh llama
#   bash scripts/run_cross_backbone_select.sh exaone          # needs HF license accept
#   bash scripts/run_cross_backbone_select.sh all             # llama then exaone (qwen skip if done)
#   MIN_FREE_GB=32 CUDA_VISIBLE_DEVICES=0 bash scripts/run_cross_backbone_select.sh llama
#
# After all three:
#   python3 scripts/summarize_cross_backbone.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-32}"
TARGET="${1:-llama}"

wait_gpu() {
  "$PY" - <<PY
import os, time, torch
need = float(os.environ.get("MIN_FREE_GB", "32"))
assert torch.cuda.is_available()
while True:
    free = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"[gpu] free_gb={free:.1f} need={need}", flush=True)
    if free >= need:
        break
    time.sleep(60)
print("[gpu] ready", torch.cuda.get_device_name(0), flush=True)
PY
}

run_family() {
  local family="$1"
  local cfgdir="configs/${family}"
  local out
  case "$family" in
    qwen) out="outputs/qwen35_9b" ;;
    llama) out="outputs/llama31_8b" ;;
    exaone) out="outputs/exaone30_7p8b" ;;
    *) echo "unknown family: $family"; exit 1 ;;
  esac
  mkdir -p "$out"
  local log="$out/cross_backbone.log"

  if [[ -f "$out/eval_direction_i_ii/report.json" ]]; then
    echo "[skip] $family already has report.json"
    return 0
  fi

  # EXAONE license gate
  if [[ "$family" == "exaone" ]]; then
    if ! "$PY" - <<'PY'
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", trust_remote_code=True)
    print("exaone_access_ok")
except Exception as e:
    print("EXAONE_ACCESS_DENIED:", e)
    raise SystemExit(2)
PY
    then
      echo "EXAONE gated: accept license at https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct then rerun."
      return 2
    fi
  fi

  wait_gpu
  {
    echo "=== FAMILY=$family OUT=$out $(date -Iseconds) ==="
    "$PY" scripts/train_stage1_en.py --config "$cfgdir/stage1_en.yaml"
    "$PY" scripts/run_stage2_gates.py --config "$cfgdir/stage2_gates.yaml"
    "$PY" - <<PY
import json, sys
from pathlib import Path
p = Path("$out/stage2_gates/gates.json")
g = json.loads(p.read_text())["gates"]
dec = {k: v["decision"] for k, v in g.items()}
print("gates", dec)
if all(v == "share" for v in dec.values()):
    print("FATAL all-share", file=sys.stderr); sys.exit(2)
PY
    # Fair protocol: same Stage3 recipe as original SELECT (auto + gates), not v2 soft_share
    "$PY" scripts/train_stage3_ko.py \
      --config "$cfgdir/stage3_ko.yaml" \
      --init_mode auto \
      --gates_file "$out/stage2_gates/gates.json" \
      --en_replay_every 0 \
      --lora_anchor_weight 0.0 \
      --output_dir "$out/stage3_ko"

    "$PY" scripts/train_stage3_ko.py \
      --config "$cfgdir/stage3_ko.yaml" \
      --init_mode relearn \
      --gates_file "$out/stage2_gates/gates.json" \
      --en_replay_every 0 \
      --lora_anchor_weight 0.0 \
      --output_dir "$out/baseline_ko_scratch"

    "$PY" scripts/train_stage3_ko.py \
      --config "$cfgdir/stage3_ko.yaml" \
      --init_mode share \
      --gates_file "$out/stage2_gates/gates.json" \
      --en_replay_every 0 \
      --lora_anchor_weight 0.0 \
      --output_dir "$out/baseline_blind_share"

    "$PY" scripts/eval_direction_i_ii.py --config "$cfgdir/eval_direction_i_ii.yaml"
    echo "=== DONE $family $(date -Iseconds) ==="
  } 2>&1 | tee -a "$log"
}

case "$TARGET" in
  qwen|llama|exaone)
    run_family "$TARGET"
    ;;
  all)
    # Qwen main tree may already exist
    run_family qwen || true
    run_family llama
    run_family exaone || true
    "$PY" scripts/summarize_cross_backbone.py || true
    ;;
  *)
    echo "Usage: $0 {qwen|llama|exaone|all}"
    exit 1
    ;;
esac
