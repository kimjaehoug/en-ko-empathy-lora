# Cursor Agent Handoff — SELENE Qwen3.5-9B Experiments (NVIDIA L40S)

Paste this entire file as the first message in a **new Cursor chat** opened on the repo root  
`/Users/gimjaehong/Documents/en-ko-empathy-lora`  
(or the same repo on the L40S machine). Respond in Korean unless asked otherwise.

---

## Role

You are implementing and running the **SELENE / Factor-LoRA SELECT** experiment pipeline for the IEEE TAFFC draft.  
Paper draft: `Computer_Society_LaTeX_template/factor_lora_select.tex`  
Do **not** invent GPT-2 results as paper numbers. Replace placeholders only after real Qwen runs.

## Hardware target

- **GPU:** NVIDIA **L40S**, **48GB** VRAM
- **Preferred training:** frozen backbone + **bf16 LoRA** (PEFT). **No QLoRA / 4-bit required.**
- Comfortable defaults on L40S: `batch_size=2~4`, `grad_accum=4~8`, `max_length=512` (raise if needed), LoRA `r=16` (try `r=32` if stable), optional gradient checkpointing only if you push context/batch hard
- Full fine-tuning of 9B is still **out of scope** (paper freezes backbone); stick to LoRA + heads
- Dev Mac (Apple M4) has Ollama `qwen3.5:9b` GGUF for chat only — **not** for PEFT. Training needs HuggingFace `Qwen/Qwen3.5-9B` via `transformers` + `peft` on the L40S box

## Goal (this agent)

Ship a **reproducible Qwen3.5-9B bf16 LoRA pipeline** that fills Direction I / II / ablation tables with real metrics, and fix gate collapse so SELECT ≠ Blind share.

### Success criteria

1. Configs point to Qwen3.5-9B (not `gpt2`).
2. Stage1 → Stage2 → Stage3 → baselines → `eval_direction_i_ii` runs end-to-end on L40S.
3. Stage2 gates are **not** all-`share` on a proper subsample (expect A share-ish, S/R relearn under paper prior).
4. `outputs/eval_direction_i_ii/report.json` and ablation report written with Qwen paths.
5. One shell entrypoint, e.g. `scripts/run_qwen_full.sh`, documents env + steps.
6. Brief note in chat: VRAM peak (approx), wall time, final Acc numbers.

### Out of scope (unless asked)

- Factor-separated LoRA **banks** (A/S/R multi-adapter) — paper Method mentions them; **first ship single LoRA + gate-conditioned init** (`share` / `relearn` / `suppress` / majority or `affect_priority`). Document factor banks as follow-up.
- Rewriting the whole paper; only update numbers/paths if user asks.
- Commits / push unless user explicitly asks.
- QLoRA / bitsandbytes — **do not add** unless something unexpected forces it.

---

## Project facts (do not rediscover from scratch)

### Method

1. **Stage1 EN:** LoRA + affect head on EmpatheticDialogues (`data/processed/ed_*.jsonl`).
2. **Stage2 gates:** KO probes + EN–KO / KoED signals → `{share, relearn, suppress}` per factor (`scripts/run_stage2_gates.py` → `gates.json`).
3. **Stage3 KO:** Gate-conditioned LoRA init + multitask A/S/R on AI Hub (`data/processed/aihub_*.jsonl`).
4. **Eval:** Dir I = KO A/S/R Acc; Dir II = return-to-EN emotion/LM after KO.

### Theory / formalization (paper)

- A/C/S/R is a **working factorization**, not a uniqueness theorem.
- Props P1–P4 are empirical; Lemmas are about Γ monotonicity/stability and \(\mathcal{I}\circ\Gamma\) init.
- See Problem Formulation in `factor_lora_select.tex`.

### Known bugs / gaps

| Issue | Detail |
|-------|--------|
| Backbone | Stage YAMLs hardcode `gpt2`. `configs/backbone_selene.yaml` is docs-only. |
| Dtype / device | No explicit bf16 / CUDA `device_map` path in `src/models/stage1.py`, `stage3.py` (was Mac MPS + gpt2). |
| Gates | GPT-2 smoke → **all share** (`outputs/stage2_gates/gates.json`). SELECT ≡ Blind. Relation probe Acc=1.0 on tiny samples is unreliable. |
| Stage3 init | **Single LoRA + majority vote** of gates (`src/models/stage3.py` `build_stage3_lm`), not per-factor banks. |
| Schedule | `max_steps: 50` smoke only — paper runs need full epochs or large step budgets. |
| Deps | CUDA torch on L40S; `peft`, `transformers`, `accelerate` — **bitsandbytes not required**. |
| Disk | Ensure space for HF weights (~20GB+) + checkpoints on the L40S machine. |

### Data (already processed — do not re-download unless missing)

- `data/processed/ed_{train,valid,test}.jsonl`
- `data/processed/aihub_{train,valid}.jsonl` (A/S/R labels)
- `data/processed/koed_test.jsonl` (eval / culture signal; **not** for Stage3 train)
- `data/processed/kor_ed_*.jsonl` (translate-train baseline; wire if time)

### Key scripts

- `scripts/train_stage1_en.py`
- `scripts/run_stage2_gates.py`
- `scripts/train_stage3_ko.py` (`--init_mode`, `--gates_file`, `--lm_only`)
- `scripts/eval_direction_i_ii.py`
- `scripts/run_baselines_direction_i_ii.sh`
- `scripts/run_ablation_smoke.sh`

### Output layout (keep)

```
outputs/qwen35_9b/          # preferred root for this run
  stage1_en/
  stage2_gates/gates.json
  stage3_ko/                # SELECT
  baseline_ko_scratch/
  baseline_blind_share/
  eval_direction_i_ii/report.json
  ablation/...
```

Do **not** overwrite old `outputs/stage1_en` GPT-2 artifacts; use a Qwen-prefixed tree or new config `output_dir`s.

---

## Implementation plan (execute in order)

### Phase A — Infra (before long train)

1. Confirm CUDA torch on L40S: `torch.cuda.is_available()`, device name contains `L40S`.
2. Extend `build_lora_lm` / Stage3 loader for Qwen:
   - Load with `torch_dtype=torch.bfloat16` (or `dtype=torch.bfloat16` per installed transformers API)
   - Place on CUDA (`.to("cuda")` or `device_map="auto"` — either fine on 48GB)
   - LoRA targets: `q_proj,k_proj,v_proj,o_proj` (optionally add `gate_proj,up_proj,down_proj` for stronger adapt)
   - **Do not** wire 4-bit / QLoRA
3. New configs under `configs/qwen/`:
   - `stage1_en.yaml`, `stage2_gates.yaml`, `stage3_ko.yaml`, `eval_direction_i_ii.yaml`
   - `model_name: Qwen/Qwen3.5-9B` (verify Hub ID; if renamed, fix and note)
   - `max_steps: null` or large; `num_epochs: 1` first pass, then 2–3 if time
   - `batch_size: 2` or `4`, `grad_accum: 4` or `8`, `max_length: 512`
   - `lora_r: 16`, `lora_alpha: 32`
   - Stage2: raise `max_*_samples` (e.g. 2k–8k), keep S/R **relearn prior** unless probe ≥ share threshold; tune thresholds so A can share while S/R relearn
4. Smoke: `max_steps=20` Stage1 on CUDA — confirm adapter saves and VRAM headroom.

### Phase B — Main train

1. Stage1 full (or 1 epoch) → `outputs/qwen35_9b/stage1_en`
2. Stage2 gates → inspect `gates.json`; **fail the run** if all gates are share without justification; fix thresholds/priors/samples
3. Stage3 SELECT (`init_mode=auto` + gates)
4. Baselines: KO-scratch (`relearn`), Blind share (`share`)
5. Optional: EN-only zero-shot Dir I; translate-train (`kor_ed`) if time
6. `eval_direction_i_ii.py` with Qwen config; write `report.json`

### Phase C — Prop / ablation (after main numbers)

1. Prop1–2: probe Acc + cross-factor / Cramér’s V on frozen Stage1 encoder (script if missing: `scripts/eval_props.py`)
2. Prop3: ordering \(s_A \gg s_S, s_R\)
3. Prop4: SELECT vs Blind Dir I / |\Delta Dir II|
4. Ablations A1–A5 matching paper table (w/o Stage1, w/o gates, LM-only, Stage1-only, all-relearn)

### Phase D — Report back

- Paste key metrics table (Dir I A/S/R Acc; Dir II Δ; gates)
- List config paths and commands used
- Note any Hub ID / tokenizer / chat-template quirks for Qwen3.5
- Do **not** commit unless asked

---

## Constraints / quality

- Prefer editing existing scripts over parallel forks.
- Match existing JSONL schema and head interfaces.
- Keep runs deterministic: `seed: 42`, log `run_meta.json` (model, steps, VRAM, git commit hash).
- If HF download fails, retry with token; do not silently fall back to `gpt2`.
- Paper citation rule (if touching `.tex`): Q1 only — but this agent should focus on **code + experiments**, not paper rewrite.
- After any `.tex` edit, run `Computer_Society_LaTeX_template/compile.sh`.

---

## Suggested first commands (on L40S machine)

```bash
cd /path/to/en-ko-empathy-lora
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
# then implement Phase A, smoke 20 steps, then full pipeline
```

## Definition of done checklist

- [ ] bf16 LoRA loads Qwen3.5-9B on L40S (no 4-bit)
- [ ] Stage1 checkpoint saved
- [ ] Stage2 gates.json with mixed share/relearn (not all-share)
- [ ] Stage3 SELECT + two baselines trained
- [ ] `report.json` Dir I/II
- [ ] Commands documented in `scripts/run_qwen_full.sh` (or README section)

Start with Phase A. If the Hub model ID for Qwen3.5-9B differs from `Qwen/Qwen3.5-9B`, resolve it first and update configs accordingly.
