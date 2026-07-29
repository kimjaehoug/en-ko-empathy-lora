# A/C/S/R 축 — 이론 기준과 참고 논문 (Q1 / Top-tier)

축은 데이터 클러스터링으로 만든 것이 아니라 **아래 문헌의 구성 개념을 사전 정의**한 것입니다.

| 축 | 이론에서 말하는 것 | 우리 데이터 라벨 | 핵심 참고 |
|----|-------------------|------------------|-----------|
| **A Affect** | 정서적 공감 / Empathic Concern (느끼기) | ED 32감정, AI Hub 6감정 | Davis (1983) *JPSP* Q1; Rashkin ED (ACL’19) |
| **C Cognition** | 관점수용 / Perspective-Taking, 상황 평가 | Situation 텍스트 (proxy) | Davis (1983) PT; Rashkin ED situation |
| **S Strategy** | 돕는 기술·의도 (탐색→위로→행동), 감정 미러링 이상 | 조언·격려·위로·동조 | Liu et al. ESC (ACL’21); Hill Helping Skills는 ESC가 NLP로 정량화 |
| **R Relation** | display rules·사회맥락 감정조절 + 한국어 관계 규범 | AI Hub 관계 7종 | Matsumoto et al. (2008) *JPSP* Q1; Lee et al. KoED (AACL’25) |

## 문헌 → 축 매핑 (한 줄)

1. **Davis 1983 (JPSP)** — 공감은 단일 차원이 아님 → A(EC)와 C(PT)를 **분리**할 근거  
   DOI: https://doi.org/10.1037/0022-3514.44.1.113  
2. **Rashkin et al. 2019 (ACL)** — ED: 감정(A) + situation(C) + 공감 응답 생성  
   https://aclanthology.org/P19-1534.pdf  
3. **Liu et al. 2021 (ACL)** — ESC: 전략(S)이 공감 품질을 좌우; Hill helping skills의 대화 시스템 정량화  
   https://aclanthology.org/2021.acl-long.269.pdf  
4. **Matsumoto et al. 2008 (JPSP)** — 문화·감정조절·사회적 맥락 → R(관계/표현 규범)  
   DOI: 10.1037/0022-3514.94.6.925  
5. **Lee et al. 2025 KoED (AACL)** — 한국어에서 문화·관계 공감 갭 → R의 한국어 특수성  
   https://aclanthology.org/2025.ijcnlp-long.44.pdf  

## 보조 (이미 초안에 있는 것)

- Sabour et al. EmoBench (ACL’24) — 다층 감정지능 평가  
- Amin et al. (IEEE TAFFC) — LLM 감성 과제 전반  

## 의도적으로 안 쓴 것 (비-Q1 / 책)

- Hall *Beyond Culture* (단행본)  
- Welivita COLING intent taxonomy → S는 ESC(ACL)로 대체  
- Clara Hill *Helping Skills* 원서 → ESC가 인용·구현하므로 ESC만 cite  
