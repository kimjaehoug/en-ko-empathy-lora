# Factor-LoRA SELECT — IEEE LaTeX draft

메인 원고: [`factor_lora_select.tex`](factor_lora_select.tex)  
미리보기 PDF: [`factor_lora_select.pdf`](factor_lora_select.pdf)

## 저자 (JAMA Network Open 제출본 기준)

- Jaehong Kim
- Seohyeon Yoo, PhD
- Jaehyuk Cho, PhD (corresponding: chojh@jbnu.ac.kr)
- Affiliation: Department of Software Engineering, Jeonbuk National University, Jeonju, Republic of Korea

## 컴파일 (TinyTeX)

이 워크스페이스에는 **TinyTeX**가 설치되어 있습니다 (`~/Library/TinyTeX`).

```bash
./compile.sh
# 또는
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
pdflatex factor_lora_select.tex && pdflatex factor_lora_select.tex
```

Cursor에서 미리보기:
1. `factor_lora_select.tex` 열기
2. LaTeX Workshop 확장 설치 시 `Cmd+Option+B` 빌드 → PDF 탭
3. 또는 생성된 `factor_lora_select.pdf`를 직접 열기

프로젝트 설정: `../.vscode/settings.json` (TinyTeX 경로 지정)
