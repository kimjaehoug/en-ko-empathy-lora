# EN–KO Empathy LoRA

IEEE TAC 방향 연구 워크스페이스.  
EN 백본 유지 + divergence-gated language/factor LoRA + 합성 어댑터.

## Theory axes (사전정의)

| Axis | Theory anchor | Measure |
|------|---------------|---------|
| A Affect | affective empathy | emotion Acc/F1 |
| C Cognition | cognitive empathy / appraisal | situation/cause proxy |
| S Strategy | Welivita intents + ESC + AI Hub | Macro-F1 (main) |
| R Relation | display rules + KO honorific/DCT | relation + honorific match |

## Data layout

```
data/raw/
  empathetic_dialogues/   # EN
  KoED/                   # EN–KO parallel eval
  kor_empathetic_dialogues/  # KO translation baseline
  aihub_empathy/          # KO main (manual from AI Hub)
data/processed/
```

## Setup

```bash
pip install datasets huggingface_hub
python scripts/download_public_data.py
```

AI Hub 공감형 대화: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71305  
신청·다운로드 후 `data/raw/aihub_empathy/` 에 압축 해제.
