# Kospi Machine Learning (GVIF7) - 퀀트팀 프로젝트

이 프로젝트는 다양한 거시경제(Macro), 시장(Market), 가격(Price), 거래량(Volume) 데이터를 활용하여 **KOSPI 지수 방향성을 예측**하고, 이를 기반으로 **트레이딩 전략을 백테스팅**하여 최적의 수익률을 달성하는 것을 목표로 합니다.

---

## 🚀 1. 프로젝트 개요 (What We Did)

우리 퀀트팀은 단순한 모델 학습을 넘어, 실전 트레이딩에 적용할 수 있는 강건한 모델을 만들기 위해 데이터 전처리부터 모델 튜닝까지 광범위한 실험을 진행했습니다.

- **데이터 수집 및 전처리 (`src/data`, `src/preprocessing`)**:
  - 정상성(Stationarity) 확보를 위한 ADF 테스트 및 변환 (Log 1st diff 등).
  - 모델의 과적합을 방지하기 위한 VIF(다중공선성) 기반 불필요한 Feature 사전 제거.
- **다양한 모델 아키텍처 실험 (`src/models`)**:
  - 주력 모델인 **XGBoost**를 적용하고 Walk-forward 백테스팅을 진행했습니다.
- **최종 수익률 극대화 튜닝 (`src/퀀트팀_최종_성적`)**:
  - **데이터 불균형 해소**: 라벨(상승/하락)의 불균형을 맞추는 실험.
  - **Feature 최적화**: 중요도 기반 하위 Feature 반복 제거.
  - **시그널 임계값(Threshold) 조절**: 매수/매도 시그널의 확률 밑(Base)을 변경하여 리스크 관리 및 수익률 극대화.

---

## 📊 2. 주요 실험 및 결과 (Results & Reports)

### 2.1. 데이터 불균형 실험 (`src/퀀트팀_최종_성적/데이터불균형실험`)
- **내용**: 타겟 변수(KOSPI 상승/하락)의 클래스 비율을 맞춘 데이터(균형)와 원본 데이터(불균형) 간의 성능을 비교했습니다. (시드별: 11, 22, 33, 42)
- **결과**:
  - 각 시드별 폴더 내 `수익률.txt` 및 시각화 그래프(`*_그래프.png`)를 통해 불균형/균형 데이터에 따른 백테스트 누적 수익률 차이를 확인할 수 있습니다.
  - *결과물 위치: `src/퀀트팀_최종_성적/데이터불균형실험/`*

### 2.2. Feature 최적화 및 변수 지우기 (`src/퀀트팀_최종_성적/최종_featrue지우기`)
- **내용**: 모델이 뽑아낸 Feature Importance(중요도)를 바탕으로, 중요도가 낮은 변수를 단계별로(1차~5차) 제거하며 백테스팅 성능의 변화를 추적했습니다.
- **결과**:
  - 지속적인 변수 제거 결과, **4차 제거 시점이 Best 모델**로 판명되었습니다. (`4차가 best로보임.txt`)
  - *결과물 위치: `src/퀀트팀_최종_성적/최종_featrue지우기/`*

### 2.3. 임계값(Threshold/밑) 변화 실험 (`src/퀀트팀_최종_성적/최종본_밑변화주기`)
- **내용**: 모델이 예측하는 상승 확률의 임계값(밑: 1.7, 1.9, 2, 3, e 등)을 조정하여, 얼마나 확실할 때 매수/매도를 진행할지 실험했습니다.
- **결과**:
  - 균형 및 불균형 데이터에 대해 각 임계값별로 누적 수익률을 계산한 CSV 파일(`균형_밑_1.9.csv`, `불균형_밑e.csv` 등)이 도출되었습니다.
  - 이를 통해 보수적/공격적 트레이딩의 성과를 직접 비교할 수 있습니다.

### 2.4. 퀀트팀 최종 성적표
- 전체 데이터를 활용한 최종 백테스팅 시뮬레이션(Walk-forward) 결과입니다.
- 주요 파일: `data/xgboost_data/kospi_final_daily_results_3차.csv`, `data/xgboost_data/kospi_final_daily_results_4차.csv`

**[3차 변수 제거 모델 혼동행렬 및 수익률 요약]**
- **TP(실제 상승 예측 성공)**: 252회
- **TN(실제 하락 예측 성공/방어)**: 112회
- **FP(상승으로 잘못 예측)**: 171회
- **FN(하락으로 잘못 예측)**: 126회
- **전체 기간 정확도(Accuracy)**: 약 55.07% (364 / 661)
- **백테스트 누적 수익률**: Long-Only **201.26%**, Long-Short **174.51%**

**[4차 변수 제거 모델 (Best) 혼동행렬 및 수익률 요약]**
- **TP(실제 상승 예측 성공)**: 267회
- **TN(실제 하락 예측 성공/방어)**: 113회
- **FP(상승으로 잘못 예측)**: 170회
- **FN(하락으로 잘못 예측)**: 111회
- **전체 기간 정확도(Accuracy)**: 약 57.49% (380 / 661)
- **백테스트 누적 수익률**: Long-Only **114.20%**, Long-Short **37.37%**

---

## 💻 3. 코드 실행 가이드 (How to Run)

각종 분석 리포트 생성 및 모델 학습 코드는 아래와 같이 실행할 수 있습니다.

### 📈 EDA 및 리포트 생성
**1. 다중공선성(VIF) 및 상관관계 분석 리포트**
데이터의 상관관계와 VIF 수치를 계산하여 리포트를 생성하고, 기준치를 넘는 변수를 자동으로 제거한 데이터셋을 만듭니다.
```powershell
python src/evaluation/correlation_vif_report.py --input data/merged_data.csv --target y_KOSPI_Close
```
*출력: `outputs/correlation_vif/index.html`*

**2. 정상성(ADF) 및 자기상관(ACF/PACF) 진단**
시계열 데이터의 정상성을 확보하기 위한 추천 변환(Differencing, Log 등)을 도출합니다.
```powershell
python src/preprocessing/stationarity_autocorr_diagnostics.py --input data/merged_data.csv --target Close
```
*출력: `outputs/stationarity_autocorr/` 폴더 내 결과물*

### 🤖 머신러닝 모델 학습 및 백테스팅

**1. XGBoost Batch Backtest**
여러 조건하에 XGBoost의 백테스트를 일괄 실행합니다.
```powershell
python src/backtest/xgboost_batch_backtest.py
```

**2. 퀀트팀 최종 모델 백테스팅**
최종 최적화된 로직(Walk-forward Optuna 튜닝)과 임계값이 적용된 XGBoost 모델의 수익률을 시뮬레이션 합니다.
```powershell
python src/퀀트팀_최종모델/퀀트팀_모델_백테스팅_최종.py
```
