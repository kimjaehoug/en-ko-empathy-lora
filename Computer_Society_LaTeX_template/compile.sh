#!/usr/bin/env bash
# Compile IEEE draft with TinyTeX (no sudo / no MacTeX required)
set -euo pipefail
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
pdflatex -interaction=nonstopmode factor_lora_select.tex
pdflatex -interaction=nonstopmode factor_lora_select.tex
echo "PDF -> $ROOT/factor_lora_select.pdf"
