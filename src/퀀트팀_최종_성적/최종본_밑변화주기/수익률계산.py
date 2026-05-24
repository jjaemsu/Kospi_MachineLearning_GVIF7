import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. KOSPI 벤치마크 데이터 준비
kospi = pd.read_csv('kospi_data.csv')
kospi['Date'] = pd.to_datetime(kospi['Date'])
kospi.set_index('Date', inplace=True)
kospi.sort_index(inplace=True)

# --- [핵심 로직] 두 가지 진입/청산 기준에 따른 수익률 정의 ---
# ① 종가 기준 수익률: 어제 종가 진입 -> 오늘 종가 청산
kospi['Return_Close'] = kospi['Close'].pct_change()

# ② 시가 기준 수익률: 오늘 아침 시가 진입 -> 내일 아침 시가 청산
# (오늘 시가 대비 내일 시가의 수익률을 구하기 위해 shift(-1) 사용)
kospi['Return_Open'] = kospi['Open'].shift(-1) / kospi['Open'] - 1

# 백테스트 마지막 날은 '내일 시가' 데이터가 없으므로 '오늘 종가'에 청산했다고 가정(fillna)
kospi['Return_Open'] = kospi['Return_Open'].fillna(kospi['Close'] / kospi['Open'] - 1)

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

# 4. 공통 수수료율 설정 (0.015%)
FEE_RATE = 0.00015  

# 5. 그래프 초기화
fig, ax = plt.subplots(figsize=(14, 8))

# KOSPI 벤치마크 누적 수익률 계산
kospi_matched = kospi[kospi.index >= common_start_date].copy()
kospi_matched = kospi_matched.dropna(subset=['Return_Close', 'Return_Open'])

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
        continue
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 모델 데이터에 코스피의 두 가지 수익률 모두 결합
    df = df.join(kospi[['Return_Close', 'Return_Open']])
    df = df.dropna(subset=['Return_Close', 'Return_Open'])
    
    # 공통 포지션 및 거래 발생 여부
    df['Position'] = df['Predicted']
    df['Trade'] = df['Position'].diff().fillna(df['Position']).abs()
    df['Fee_Multiplier'] = 1 - (df['Trade'] * FEE_RATE)
    
    # --- (1) 종가 기준 (Close) 시뮬레이션 ---
    df['Gross_Mult_Close'] = 1 + (df['Position'] * df['Return_Close'])
    df['Cum_Return_Close'] = (df['Gross_Mult_Close'] * df['Fee_Multiplier']).cumprod() - 1
    
    # --- (2) 시가 기준 (Open) 시뮬레이션 ---
    df['Gross_Mult_Open'] = 1 + (df['Position'] * df['Return_Open'])
    df['Cum_Return_Open'] = (df['Gross_Mult_Open'] * df['Fee_Multiplier']).cumprod() - 1
    
    # 모델 시각화 그리기 (종가 전략은 파란 실선, 시가 전략은 붉은 점선)
    ax.plot(df.index, df['Cum_Return_Close'] * 100, label=f'{name} (Close Entry)', 
            color='blue', linewidth=2.5, linestyle='-')


# 7. 차트 마무리 설정
ax.set_title('Backtest Comparison: Market-On-Close vs Market-On-Open', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)

# 범례 위치 조정
ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()