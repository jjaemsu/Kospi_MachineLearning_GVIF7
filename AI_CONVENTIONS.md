# AI & Team Conventions for KOSPI Prediction Project

이 문서는 팀원들이 사용하는 모든 AI 어시스턴트(Gemini, Cursor, Copilot, Cline 등)와 팀원들이 공통으로 준수해야 할 프로젝트 규칙 및 가이드라인입니다.

## 1. 프로젝트 개요 (Project Overview)
- **목표:** 코스피(KOSPI) 지수의 방향성(상승/하락) 예측
- **주요 방법론:** 머신러닝 (Machine Learning) 및 딥러닝 기반 시계열 분류/회귀

## 2. AI 어시스턴트 주의사항 (AI Assistant Guidelines)

모든 AI는 코드를 작성하거나 구조를 제안할 때 다음 사항을 엄격히 준수해야 합니다:

### 2.1. 사전 코드 작성 금지 (No Preemptive Coding)
- 사용자가 명시적으로 코드 작성을 요청하기 전까지는 임의로 전체 코드를 작성하거나 완성하지 마세요.
- 질문이나 아이디어 단계에서는 로직 설계, 데이터 흐름, 구조에 대한 논의에 집중하세요.

### 2.2. 폴더 구조 준수 (Strict Architecture Adherence)
- 기존에 정의된 프로젝트 폴더 구조(`src/features`, `src/models`, `src/backtest` 등)를 반드시 따르세요.
- 각 모듈은 독립적으로 실행 가능하고 재사용성이 높도록 모듈화(Modularization)되어야 합니다.

### 2.3. 실험 기록 및 재현성 (Reproducibility)
- 모든 모델 학습 및 평가 결과는 `experiments/` 또는 `outputs/`에 기록되는 구조를 염두에 두고 로직을 설계하세요.
- 랜덤 시드(Random Seed) 고정은 필수입니다.

---
**AI 프롬프트 지침:** AI 어시스턴트는 이 파일을 컨텍스트로 읽어들인 후, 코스피 방향성 예측이라는 도메인 특성에 맞춰 답변을 생성해야 합니다.