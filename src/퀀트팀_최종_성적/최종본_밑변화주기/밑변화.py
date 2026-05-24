import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. KOSPI 벤치마크 데이터 준비
kospi = pd.read_csv('kospi_data.csv')
kospi['Date'] = pd.to_datetime(kospi['Date'])
kospi.set_index('Date', inplace=True)
kospi.sort_index(inplace=True)

# --- [정교화된 핵심 로직] 진입/청산 시점과 수익률 매핑 정상화 ---
# 오늘(t일) 장 마감 후 발생한 예측 포지션(Predicted)에 대한 올바른 미래 수익률 정의

# ① 종가 기준 수익률 (Market-On-Close): 오늘 종가 진입 -> 내일 종가 청산
# t일 종가에 매수하여 t+1일 종가에 매도하므로, t+1일의 Close pct_change를 t일 포지션에 매칭해야 함
kospi['Return_Close'] = kospi['Close'].pct_change().shift(-1)

# ② 시가 기준 수익률 (Market-On-Open): 내일 아침 시가 진입 -> 모레 아침 시가 청산
# t일 장 마감 후 예측했으므로 가장 빠른 시가 진입은 t+1일 시가이며, 청산은 t+2일 시가임
# 즉, (t+2일 시가 / t+1일 시가) - 1 성과를 t일 포지션에 매칭해야 함
kospi['Return_Open'] = kospi['Open'].shift(-2) / kospi['Open'].shift(-1) - 1

# 백테스트 마지막 구간 데이터 결측치 처리
# 마지막 날은 t+1일 종가 데이터가 없으므로 당일 종가 정산 가정
kospi['Return_Close'] = kospi['Return_Close'].fillna(0)
# 마지막에서 두 번째 날은 t+2일 시가가 없으므로 t+1일 종가 청산으로 방어적 처리
kospi['Return_Open'] = kospi['Return_Open'].fillna(kospi['Close'].shift(-1) / kospi['Open'].shift(-1) - 1).fillna(0)


# 2. 분석할 모델 파일 목록 정의
files = {
    "Imbalanced Base 1.7 (불균형)": "불균형_밑1.7.csv",
}

# 3. 공통 시작일 동적 탐색
start_dates = []
for file in files.values():
    try:
        temp_df = pd.read_csv(file)
        start_dates.append(pd.to_datetime(temp_df['Date']).min())
    except FileNotFoundError:
        pass
common_start_date = max(start_dates) if start_dates else kospi.index.min()

# 4. 현실적인 거래 비용 설정 (수수료 0.015% + 슬리피지/호가갭 보수적 반영 = 총 0.05%)
# 종가/시가 동시호가 체결 시 발생하는 미체결 및 호가 밀림을 방어하기 위해 0.05%~0.1% 설정을 권장합니다.
FEE_RATE = 0.00

# 5. 그래프 초기화
fig, ax = plt.subplots(figsize=(14, 8))

# KOSPI 벤치마크 누적 수익률 계산
kospi_matched = kospi[kospi.index >= common_start_date].copy()

# 벤치마크 복리 계산 (첫 진입 수수료 1회 지불 반영)
kospi_matched['Cum_Return_Close'] = ((1 + kospi_matched['Return_Close']).cumprod() * (1 - FEE_RATE)) - 1
kospi_matched['Cum_Return_Open'] = ((1 + kospi_matched['Return_Open']).cumprod() * (1 - FEE_RATE)) - 1

# KOSPI 라인 그리기 (종가는 검은색 실선, 시가는 회색 점선)
ax.plot(kospi_matched.index, kospi_matched['Cum_Return_Close'] * 100, 
        label='KOSPI (Close B&H)', color='black', linewidth=2.5, linestyle='-')
ax.plot(kospi_matched.index, kospi_matched['Cum_Return_Open'] * 100, 
        label='KOSPI (Open B&H)', color='gray', linewidth=2, linestyle='--')


# 6. 각 모델별 성과 계산 (Open vs Close 동시 시뮬레이션)
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for (name, file), color in zip(files.items(), colors):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"오류: {file} 파일을 찾을 수 없습니다.")
        continue
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 모델 데이터에 코스피의 수정된 두 가지 미래 수익률 결합
    df = df.join(kospi[['Return_Close', 'Return_Open']])
    df = df.dropna(subset=['Return_Close', 'Return_Open'])
    
    # 공통 포지션 및 거래 발생 여부 (포지션 변동 시에만 수수료 차감)
    df['Position'] = df['Predicted']
    df['Trade'] = df['Position'].diff().fillna(df['Position']).abs()
    df['Fee_Multiplier'] = 1 - (df['Trade'] * FEE_RATE)
    
    # --- (1) 종가 기준 (Close) 시뮬레이션 ---
    # 오늘 포지션(t) 카드와 내일 실제 일어날 종가 수익률(t+1)의 곱연산
    df['Gross_Mult_Close'] = 1 + (df['Position'] * df['Return_Close'])
    df['Cum_Return_Close'] = (df['Gross_Mult_Close'] * df['Fee_Multiplier']).cumprod() - 1
    
    # --- (2) 시가 기준 (Open) 시뮬레이션 ---
    # 오늘 포지션(t) 카드와 내일 시가 매수~모레 시가 매도 수익률의 곱연산
    df['Gross_Mult_Open'] = 1 + (df['Position'] * df['Return_Open'])
    df['Cum_Return_Open'] = (df['Gross_Mult_Open'] * df['Fee_Multiplier']).cumprod() - 1
    
    # 모델 시각화 그리기 (종가 전략은 파란 실선, 시가 전략은 빨간 점선으로 비교 가능)
    ax.plot(df.index, df['Cum_Return_Close'] * 100, label=f'{name} (Close Entry)', 
            color='blue', linewidth=2.5, linestyle='-')
    ax.plot(df.index, df['Cum_Return_Open'] * 100, label=f'{name} (Open Entry)', 
            color='red', linewidth=2, linestyle='--')


# 7. 차트 마무리 설정
ax.set_title('Corrected Backtest Comparison: Market-On-Close vs Market-On-Open', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)

# 범례 위치 조정
ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()