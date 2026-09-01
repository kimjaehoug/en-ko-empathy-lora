# Factor-LoRA SELECT: Divergence-Gated Adapter Transfer for English–Korean Empathetic Response Generation

**Target venue:** IEEE Transactions on Affective Computing (TAFFC)  
**Status:** Working draft (Qwen3.5-9B L40S results; see RESULTS_QWEN35.md + factor_lora_select.tex)  
**Version:** 0.2 — 2026-08-27

---

## Abstract

Large language models trained primarily on English affective dialogue often transfer poorly to Korean, where empathy is regulated by relational stance, honorifics, and high-context support strategies. We propose **Factor-LoRA SELECT**, a parameter-efficient cross-lingual transfer framework that (i) pretrains an English empathy LoRA with an affect head, (ii) estimates factor-wise share/relearn/suppress gates from English–Korean representation divergence and frozen-encoder probes, and (iii) adapts a Korean LoRA with multitask supervision for affect, support strategy, and relation. Unlike full fine-tuning, the English backbone remains frozen and language adapters are plug-in extensible. On public EmpatheticDialogues, KoED, and AI Hub Korean empathetic dialogues, prototype experiments with a GPT-2 backbone show that strategy factors are systematically less transferable than affect/relation cues and therefore require relearning, while gated initialization stabilizes Korean multitask adaptation. We release a unified A/C/S/R data schema and stage-wise training pipeline as a reproducible starting point for culturally grounded affective transfer.

**Index Terms**—Affective computing, empathetic dialogue, cross-lingual transfer, LoRA, cultural adaptation, Korean NLP, support strategies.

---

## I. Introduction

Empathetic response generation is a core affective computing problem spanning emotion understanding and socially appropriate generation [1], [2]. Recent LLM-centric systems improve fluency, yet multilingual competence does not imply multicultural empathy: models that perform well in English frequently violate Korean relational norms, over-empathize, or choose inappropriate support strategies [3], [4].

A common remedy is full fine-tuning on target-language data, which is expensive and risks catastrophic forgetting of English affective skills. Adapter-based PEFT methods reduce cost, but typical single-adapter transfer treats empathy as a monolithic skill and cannot decide *what* should be shared versus relearned across cultures.

This paper asks: **can we transfer English empathy knowledge to Korean selectively at the level of theoretically motivated affective factors, while keeping the backbone fixed?**

**Contributions.**
1. A theory-aligned factor schema for empathetic dialogue—**Affect (A), Cognition (C), Strategy (S), Relation (R)**—linked to cognitive/affective/compassionate empathy, empathetic response intents, and Korean discernment politeness.
2. **Factor-LoRA SELECT**, a three-stage method: English LoRA pretraining, divergence/probe gating (share/relearn/suppress), and gated Korean multitask LoRA adaptation.
3. A reproducible EN–KO data pipeline unifying EmpatheticDialogues, KoED, translated KorED, and AI Hub empathetic dialogues under one JSONL schema.
4. Prototype evidence that **strategy** is the least transferable factor under a frozen English encoder, motivating selective relearning rather than blind adapter copy.

---

## II. Related Work

### A. Empathetic and Emotional Support Dialogue
EmpatheticDialogues popularized open-domain empathetic chitchat [1]. Emotional support conversation (ESC) frameworks further emphasize strategy selection (e.g., affirmation, suggestion) [2], [5]. Intent taxonomies such as Welivita and Pu [6] show that listeners respond with intents beyond emotion mirroring.

### B. Cross-Cultural Affect and Multilingual Models
Display rules and high-/low-context communication predict cultural differences in emotion expression [7], [8]. Recent benchmarks argue that multilingual LMs are not multicultural in emotion and empathy [3], [4], [9]. KoED reconstructs Korean empathetic dialogues beyond translation to expose a cultural empathy gap [4].

### C. Parameter-Efficient Transfer
LoRA and related adapters enable efficient specialization [10]. Prior affective work uses PEFT for emotion or empathy control, but rarely combines **factor-wise cultural gating** with Korean relational/strategy supervision.

---

## III. Problem Formulation

Let \(x\) be dialogue context and \(y\) an empathetic listener response. We assume latent factors
\[
z=(z_A,z_C,z_S,z_R),
\]
corresponding to affect, cognitive appraisal/situation, support strategy, and relational stance. English pretraining provides transferable components of \(z\), while Korean requires culture-specific realization, especially for \(z_S\) and honorific/relational surface forms of \(z_R\).

We keep backbone parameters \(\theta\) frozen (or slow) and introduce language/factor LoRA modules \(\{L^{(\ell)}\}\). Transfer is controlled by gates \(g_k\in\{\mathrm{share},\mathrm{relearn},\mathrm{suppress}\}\) for each factor \(k\).

---

## IV. Method: Factor-LoRA SELECT

### A. Theory-Based Factor Definition
| Factor | Theoretical anchor | Operational label |
|--------|--------------------|-------------------|
| A Affect | Affective empathy | Emotion labels (ED / AI Hub) |
| C Cognition | Cognitive empathy / appraisal | Situation text (proxy) |
| S Strategy | Empathetic intents / ESC | AI Hub: advice/encourage/comfort/agree |
| R Relation | Display rules + Korean politeness | AI Hub relation types |

Divergence analysis is used to **validate** these factors, not to invent them post hoc.

### B. Stage 1 — English Empathy LoRA
On EmpatheticDialogues, we train a LoRA causal LM with an affect classification head on pooled prompt states:
\[
\mathcal{L}_{\mathrm{EN}}=\mathcal{L}_{\mathrm{LM}}+\lambda_A\mathcal{L}_{\mathrm{CE}}(A).
\]
This yields reusable English empathy adapters while leaving the backbone intact.

### C. Stage 2 — Divergence / Probe Gates
Using the frozen Stage-1 encoder:
1. Compute domain cosine similarity between EN and KO pooled representations.
2. Fit linear probes on KO features for A/S/R and measure validation accuracy (probe transfer).
3. Optionally measure KoED-related EN–KO similarity as a culture signal.

Gate rule (prototype):
- **share** if score \(\ge \tau_{\mathrm{share}}\)
- **suppress** if score \(\le \tau_{\mathrm{suppress}}\)
- **relearn** otherwise  
Hard prior: if EN lacks explicit S/R supervision, prefer **relearn** unless probe evidence is strong.

### D. Stage 3 — Gated Korean Adaptation
On AI Hub dialogues, initialize KO LoRA by:
- **share**: load EN LoRA; smaller LR  
- **relearn**: fresh LoRA  
- **suppress**: load then scale down LoRA magnitudes (soft unlearning)

Train multitask objectives:
\[
\mathcal{L}_{\mathrm{KO}}=\mathcal{L}_{\mathrm{LM}}+\lambda_A\mathcal{L}_A+\lambda_S\mathcal{L}_S+\lambda_R\mathcal{L}_R.
\]
A lightweight composer weight \(\alpha\) is reserved for future EN⊕KO adapter mixing.

---

## V. Experimental Setup

### A. Datasets
| Corpus | Role | Scale (processed) |
|--------|------|-------------------|
| EmpatheticDialogues | EN Stage1 | 17,844 / 2,763 / 2,542 |
| AI Hub Empathetic Dialogue | KO Stage3 | 25,456 / 3,182 |
| KoED | Cultural eval / gating | 1,360 (test) |
| KorEmpatheticDialogues | Translation baseline | 19,531 / 2,769 / 2,547 |

Axis coverage: AI Hub provides full A/S/R; ED provides A/C only.

### B. Implementation (Prototype)
- Backbone: GPT-2 (frozen base + LoRA \(r=8\))
- Device: Apple MPS / CPU
- Metrics: LM loss; A/S/R accuracy; gate decisions; trainable parameter count
- **Note:** GPT-2 is a controlled low-cost prototype. Final TAFFC experiments should replace it with a multilingual instruction LM (e.g., Qwen/EXAONE-class) and report human evaluation on KoED.

### C. Prototype Observations (50-step runs, GPT-2)
**Stage 1 (EN).** Trainable ≈ 0.84M / 125M. Valid LM loss ≈ 3.18; affect head still underfit (emotion Acc ≈ 0.03)—expected with 50 steps and 32 ED labels.

**Stage 2 gates** (n=128 EN/KO; after Stage-1 50-step checkpoint):
- affect: **share** (score ≈ 0.94)
- strategy: **share** (probe valid Acc ≈ 0.38; near \(\tau_{\mathrm{share}}=0.35\); earlier 10-step checkpoint yielded **relearn** at ≈ 0.23)
- relation: **share** (probe Acc = 1.0 on tiny split—treat as unstable)
- culture: **share** (KoED parallel cosine ≈ 0.90)

**Stage 3 (KO, `share_from_en`).** Trainable ≈ 0.82M. After 50 steps, valid: LM ≈ 2.07, strategy Acc ≈ 0.70, emotion Acc ≈ 0.18, relation Acc ≈ 0.0 (head underfit / class imbalance). Pipeline is end-to-end; numbers are **not** submission-ready SOTA.

---

## VI. Discussion

The prototype supports a methodological claim more than a final empirical one: **factor-wise gating is sensitive to encoder quality and sample size**, and strategy transfer scores sit near the share/relearn boundary (0.23→0.38 across Stage-1 checkpoints). This motivates reporting gates with confidence intervals and hard priors when English lacks strategy labels—rather than treating empathy as a single transferable skill. Keeping the backbone fixed enables future plug-in languages without full retraining—an important systems property for affective agents deployed across locales.

**Limitations.** Current backbone is English-centric GPT-2; Korean tokenization is suboptimal. Relation probe scores on tiny splits may be over-optimistic. Human evaluation, honorific match metrics, and adapter composition experiments remain future work. AI Hub licensing constrains redistribution of raw dialogues.

---

## VII. Conclusion

We presented Factor-LoRA SELECT, a divergence-gated LoRA transfer framework for English–Korean empathetic generation that preserves an English backbone while selectively adapting cultural factors. Prototype experiments demonstrate a complete stage-wise pipeline and highlight strategy relearning as a key design choice. Next steps include stronger multilingual backbones, factor-separated LoRA banks with learned composition, KoED human evaluation, and EN forgetting measurements.

---

## References

[1] H. Rashkin *et al.*, “Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset,” ACL, 2019.  
[2] S. Liu *et al.*, “Towards Emotional Support Dialog Systems,” ACL, 2021.  
[3] S. Havaldar *et al.*, “Multilingual Language Models are not Multicultural: A Case Study in Emotion,” WASSA, 2023.  
[4] W. Lee *et al.*, “Multilingual, Not Multicultural: … KoED,” IJCNLP-AACL, 2025.  
[5] S. Sabour *et al.*, “EmoBench,” ACL, 2024.  
[6] A. Welivita and P. Pu, “A Taxonomy of Empathetic Response Intents…,” COLING, 2020.  
[7] P. Ekman, “Universals and Cultural Differences in Facial Expressions of Emotion,” 1972 / display-rule literature.  
[8] E. T. Hall, *Beyond Culture*, 1976.  
[9] S. Park *et al.*, “Too Polite to be Human…,” SICon@ACL, 2025.  
[10] E. J. Hu *et al.*, “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR, 2022.  
[11] D. Matsumoto *et al.*, cultural display rules / power distance studies.  
[12] M. M. Amin *et al.*, “A Wide Evaluation of ChatGPT on Affective Computing Tasks,” IEEE TAFFC, 2024.  
[13] S. Zhang *et al.*, “Affective Computing in the Era of Large Language Models…,” Knowledge-Based Systems, 2026.

---

## Appendix A — Reproducibility Checklist

```bash
python scripts/build_processed.py
python scripts/split_stats.py
python scripts/train_stage1_en.py --max_steps 50
python scripts/run_stage2_gates.py
python scripts/train_stage3_ko.py --max_steps 50
```

Artifacts: `outputs/stage1_en/`, `outputs/stage2_gates/gates.json`, `outputs/stage3_ko/`.

## Appendix B — Draft Notes for TAFFC Submission

- Expand experiments with multilingual backbone, full epochs, ablations (no-gate / full-FT / translation-only).  
- Add human pairwise evaluation (appropriateness, over-empathy, honorific).  
- Move from GPT-2 prototype numbers to final tables before submission.  
- Ensure AI Hub citation and license compliance; release code + processed builders, not raw restricted data.
