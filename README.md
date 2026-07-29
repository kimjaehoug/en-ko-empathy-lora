# EN–KO Empathy LoRA — **SELENE**

IEEE TAC 방향 연구 워크스페이스.  
시스템명 **SELENE** (*Selective Empathy LoRA with EN–KO factor gatEs*)  
= frozen backbone + divergence-gated language/factor LoRA + KO multitask 적응.

## Backbone

| Role | Model | Why |
|------|-------|-----|
| **Target (비교군 맞춤)** | `Qwen/Qwen3.5-9B` | KoED peers ≈7–8B (Llama-8B, EXAONE-7.8B)에 가장 가까운 Qwen3.5 dense |
| Optional light | `Qwen/Qwen3.5-4B` | 빠른 ablation |
| Prototype (현재) | `gpt2` | 파이프라인 스모크 |

설정: `configs/backbone_selene.yaml`

## Theory axes (사전정의)

| Axis | Theory anchor | Measure |
|------|---------------|---------|
| A Affect | affective empathy | emotion Acc/F1 |
| C Cognition | cognitive empathy / appraisal | situation/cause proxy |
| S Strategy | Welivita intents + ESC + AI Hub | Macro-F1 (main) |
| R Relation | display rules + KO honorific/DCT | relation Acc |

## Pipeline (gpt2 backbone)

1. **Stage1 EN** — LoRA + affect head (`ed_*`)
2. **Stage2 Gates** — EN/KO domain cosine + KO linear probes → share/relearn/suppress
3. **Stage3 KO** — gated LoRA init + A/S/R multitask on AI Hub

## Setup

```bash
pip install -r requirements.txt
python scripts/download_public_data.py
# AI Hub under data/raw/aihub_empathy/
python scripts/build_processed.py
python scripts/split_stats.py
```

## Train

```bash
# Stage1
python scripts/train_stage1_en.py --max_steps 50

# Stage2 gates
python scripts/run_stage2_gates.py

# Stage3 KO adapt (SELECT / gated)
python scripts/train_stage3_ko.py --max_steps 50

# Baselines + Direction I/II eval
bash scripts/run_baselines_direction_i_ii.sh 50
# or:
#   python scripts/train_stage3_ko.py --init_mode relearn --output_dir outputs/baseline_ko_scratch --max_steps 50
#   python scripts/train_stage3_ko.py --init_mode share --output_dir outputs/baseline_blind_share --max_steps 50
#   python scripts/eval_direction_i_ii.py
```

**EN data:** EmpatheticDialogues → `data/processed/ed_{train,valid,test}.jsonl`  
**KO data:** AI Hub → `data/processed/aihub_{train,valid}.jsonl`

Outputs:
- `outputs/stage1_en/`
- `outputs/stage2_gates/gates.json`
- `outputs/stage3_ko/`
- `outputs/baseline_ko_scratch/`, `outputs/baseline_blind_share/`
- `outputs/eval_direction_i_ii/report.json`

## Paper draft

- **IEEE LaTeX (권장):** [`Computer_Society_LaTeX_template/factor_lora_select.tex`](Computer_Society_LaTeX_template/factor_lora_select.tex)
- Markdown 초안: [`paper/TAFFC_draft_v0.1.md`](paper/TAFFC_draft_v0.1.md)
