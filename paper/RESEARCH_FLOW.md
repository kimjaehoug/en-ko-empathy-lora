# 마더페이퍼 + 연구 흐름

## 마더페이퍼 (1개 선정)

**KoED — Lee et al., IJCNLP-AACL 2025**  
「Multilingual, not multicultural: Uncovering the cultural empathy gap in LLMs…」  
PDF: https://aclanthology.org/2025.ijcnlp-long.44.pdf

### 왜 이 논문인가
- **문제 정의**가 우리와 동일: 다국어 LLM ≠ 다문화 공감 (EN ED 잘해도 KO에서 떨어짐).
- 보고 수치로 우리 Direction I를 정당화: ED Acc 36.84 → KoED 32.69; CA에서 KO-aligned 모델 우위.
- 우리가 풀 빈칸: KoED는 **갭을 측정**하고, SELENE는 **EN→KO를 factor-gate로 메우는 방법**을 제안.

### 방법 쪽 보조 앵커 (마더는 아님)
- AdaMergeX / MAD-X / FLARE — PEFT XLT 베이스라인  
- LF-MLF — Direction II (망각) 논의  
- ESC — Strategy 축  
- Davis / Matsumoto — A/C/R 이론  

---

## 연구 흐름 (한 장)

```
[현상] EN 공감 LLM이 KO에서 문화·관계·전략을 자주 틀림
          │  (마더: KoED)
          ▼
[원인 가설] 공감은 단일 스킬이 아님 → A/C/S/R로 분해
          │  (Davis, ESC, Matsumoto + KoED)
          ▼
[기존 한계] Blind LoRA copy / language×task merge는
            “무엇을 share/relearn할지”를 공감 축으로 못 가름
          │  (MAD-X, AdaMergeX, FLARE)
          ▼
[방법] SELENE = Factor-LoRA SELECT
        Stage1 EN LoRA → Stage2 게이트 → Stage3 gated KO
          │
          ▼
[평가] Dir I: KO에서 A/S/R↑  (vs Untouched / Blind / XLT)
       Dir II: KO 후 EN 유지   (vs forgetting; LF-MLF 직관)
       Ablation: Stage/게이트/헤드 기여 분리
          │
          ▼
[백본] Qwen3.5-9B (KoED peer ~7–8B에 맞춤)
```

## 논문 표 구조 (정리 후)

| 표 | 역할 |
|----|------|
| Tab.factors | A/C/S/R 정의 |
| Tab.data | 데이터 |
| Tab.systems | 비교 시스템 한눈에 |
| Tab.dir_i | **주결과** KO A/S/R |
| Tab.dir_ii | **주결과** EN 복귀 |
| Tab.ablation | Stage별 기여 |

실험 숫자는 초안에서 TBD로 두고, 논문 문장·표 골격 먼저 고정.
