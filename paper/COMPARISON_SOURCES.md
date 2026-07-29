# Direction I/II 성능 비교용 Q1급 논문

우리 실험과 **같은 숫자를 직접 베끼는 비교**가 아니라,  
표에 넣을 **문헌 앵커 + 재구현 베이스라인** 후보입니다.

## Direction I — KO 감정/감성 정확도 (vs 무전이 KO)

| 논문 | Venue | 보고 수치 (요약) | 우리 표에서 쓰는 법 |
|------|-------|------------------|---------------------|
| **KoED** | AACL 2025 | ED 36.84→KoED 32.69 Acc; CA EXAONE 3.71 vs Llama 2.94 | “아무것도 안 한/비전이 다국어”도 KO에서 떨어짐 |
| **AdaMergeX** | NAACL 2025 | Eng-FT 31.9→33.3 Avg; vs MAD-X +8.0/+15.9 | EN-only vs 전이 방법 상한 |
| **FLARE** | ACL 2025 | vs plain LoRA +2.14 (Llama) | EN→target LoRA 융합 베이스라인 |
| **MAD-X** | EMNLP 2020 | language+task adapters | 모듈러 XLT 고전 베이스라인 |
| **ESC** | ACL 2021 | strategy 조건 생성이 인간평가 우세 | A만이 아니라 S도 보고해야 함 |

PDF:
- https://aclanthology.org/2025.ijcnlp-long.44.pdf
- https://aclanthology.org/2025.naacl-long.493.pdf
- https://aclanthology.org/2025.acl-long.1255.pdf
- https://aclanthology.org/2020.emnlp-main.617.pdf
- https://aclanthology.org/2021.acl-long.269.pdf

## Direction II — KO 갔다 온 뒤 EN 복귀 (향상 vs 망각)

| 논문 | Venue | 보고 수치 (요약) | 우리 표에서 쓰는 법 |
|------|-------|------------------|---------------------|
| **LF-MLF** | NeurIPS 2022 | NER zero-shot 60.05→68.50 (덜 잊을수록 전이↑) | return-to-source / forgetting 논의 근거 |
| **LoRA / MAD-X** | ICLR / EMNLP | 플러그인 어댑터로 소스 모듈 재사용 | EN LoRA 유지 + KO LoRA 합성 정당화 |
| **AdaMergeX** | NAACL 2025 | 태스크/언어 어댑터 분리·머지 | Blind overwrite vs modular keep-EN |

PDF:
- https://proceedings.neurips.cc/paper_files/paper/2022/file/5f9f9e4da57a94547491a39dc18f1696-Paper-Conference.pdf
- https://openreview.net/pdf?id=nZeVKeeFYf9
- https://aclanthology.org/2020.emnlp-main.617.pdf

## 우리 표에 넣을 목표 형태 (실험 후)

**Table A (Dir I, AI Hub / KoED)**  
`Untouched KO | EN-only | Blind share | MAD-X/AdaMergeX/FLARE | SELECT` × `A Acc | S Acc | R Acc | (CA)`

**Table B (Dir II, EmpatheticDialogues)**  
`EN before KO | after KO-scratch | after Blind share | after SELECT (± compose α)` × `Emotion Acc | Δ vs Stage1`

권장 저장: `paper/refs/`
