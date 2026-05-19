# KOSPI 방향성 예측 프로젝트 TODO 리스트

이 파일은 프로젝트의 진행 상황을 트래킹하는 곳입니다. 팀원과 AI 어시스턴트는 작업을 시작하기 전 이 파일을 확인하고, 작업이 완료되면 상태를 업데이트(`[ ]` -> `[x]`) 해야 합니다.

## 📋 1단계: 데이터 준비 및 탐색 (Data & EDA)
<<<<<<< Updated upstream
- [x] **[Data] 코스피 주가 및 거래량/수급 데이터 수집** (`src/data/fetch_price.py`, `src/data/fetch_volume.py`)
  - `FinanceDataReader` 및 `pykrx`를 사용하여 주가, 거래량, 투자자별 수급 데이터를 통합 수집.
- [X] **[Data] 거시경제 지표 데이터 수집 (선택)**
  - 환율, 금리, S&P500 등 외부 지표 수집.
=======
- [x] **[Data] 코스피 주가 데이터 수집 코드 작성** (`src/data/fetch_kospi.py`)
  - `FinanceDataReader` 또는 `yfinance`를 사용하여 최근 10년 코스피 지수(OHLCV) 일별 데이터 수집 후 `data/` 폴더에 CSV 저장.
- [x] **[Data] 거시경제 지표 데이터 수집 (선택)**
  - 환율, 금리, S&P500 등 외부 지표 수집 (`src/data/fetch_macro_market.py`).
>>>>>>> Stashed changes
- [ ] **[EDA] 데이터 탐색 및 시각화** (`notebooks/01_eda_basic.ipynb`)
  - 수집된 데이터 결측치 확인 및 기본 차트 시각화.

## 📋 2단계: 데이터 전처리 및 피처 엔지니어링 (Preprocessing & Features)
- [ ] **[Preprocess] 데이터 정제 및 결측치 처리** (`src/preprocessing/clean_data.py`)
  - 결측치 처리 및 날짜 포맷 통일.
- [x] **[Feature] 기술적 지표(보조지표) 파생 변수 생성** (`src/features/technical_indicators.py`)
  - 이동평균선(MA), RSI, MACD, 볼린저 밴드 등 계산 로직.
- [ ] **[Feature] 타겟 변수(Label) 생성**
  - 다음 날 종가 상승 시 `1`, 하락 시 `0`으로 방향성 라벨링.

## 📋 3단계: 모델링 (Modeling)
- [ ] **[Model] 학습/테스트 데이터 분리**
  - 시계열 특성 유지(Time-Series Split)를 통한 데이터 분리.
- [ ] **[Model] Baseline 모델 구축** (`src/models/train_baseline.py`)
  - Logistic Regression 또는 Random Forest 기반 베이스라인 확보.
- [ ] **[Model] 주력 모델 (XGBoost) 학습 코드 작성** (`src/models/train_xgboost.py`)
  - XGBoost 모델 학습 및 파라미터 세팅 로직.

## 📋 4단계: 평가 및 튜닝 (Evaluation & Tuning)
- [ ] **[Eval] 모델 성능 평가 지표 계산** (`src/evaluation/metrics.py`)
  - 정확도, F1-Score 계산 및 Confusion Matrix 출력.
- [ ] **[Tune] 하이퍼파라미터 튜닝**
  - Optuna 또는 GridSearchCV를 이용한 XGBoost 최적화.

## 📋 5단계: 백테스팅 (Backtesting)
- [ ] **[Backtest] 예측 기반 매매 시그널 생성** (`src/strategies/trading_signals.py`)
  - 1(상승) 예측 시 매수, 0(하락) 예측 시 매도 시그널 생성.
- [ ] **[Backtest] 과거 데이터 기반 수익률 시뮬레이션** (`src/backtest/run_backtest.py`)
  - 수익률 및 MDD 계산 백테스트 로직 구현.