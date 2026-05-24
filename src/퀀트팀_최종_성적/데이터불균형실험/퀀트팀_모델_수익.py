import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df_results = pd.read_csv("kospi_final_daily_results.csv")
df_data = pd.read_csv("kospi_data.csv")

# 2. 필요한 컬럼 병합 (매도를 판단하기 위해 'Predicted' 컬럼 추가 병합)
df = pd.merge(df_results[['Date', 'Predicted', 'Confidence']], df_data[['Date', 'Close']], on='Date', how='inner')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 3. 백테스트 초기 변수 설정
initial_capital = 100000000.0  # 초기 자본 1억원
cash = initial_capital         # 보유 현금
shares = 0.0                   # 보유 주식 수

# 수수료율 및 세금 설정 (코스피 실전 환경 기준)
FEE_RATE = 0.00015  # 증권사 매수 수수료 (약 0.015%)
TAX_RATE = 0.00215  # 매도 증권거래세 + 매도 수수료 (약 0.215%)

# 시각화를 위한 리스트
dates = []
total_assets = []
benchmark_assets = []

# 벤치마크(코스피 단순 보유) 비교를 위한 첫날 종가
initial_price = df['Close'].iloc[0]

# 4. 일별 매매 시뮬레이션
# (참고: 실전에서는 당일 장 마감 10분 전 현재가로 모델을 돌려 종가 부근에 체결시키는 MOC 방식을 가정합니다.)
for index, row in df.iterrows():
    pred_class = row['Predicted']
    confidence = row['Confidence']
    price = row['Close']
    
    # [매수] AI가 '상승(1)'을 예측하고, 그 확신도가 0.5 초과일 때
    THRESHOLD = 0.5
    SCALE_RANGE = 1.0 - THRESHOLD
    POWER = 2  # 2: 2차 함수(Convex), 3: 3차 함수, 1: 기존 선형(Linear)

    if pred_class == 1 and confidence > THRESHOLD:
        linear_ratio = (confidence - THRESHOLD) / SCALE_RANGE
        
        buy_ratio = min(max(linear_ratio ** POWER, 0.0), 1.0) 

        invest_amount = cash * buy_ratio
        shares += (invest_amount * (1 - FEE_RATE)) / price
        cash -= invest_amount
        
    elif pred_class == 0 and confidence > THRESHOLD:
        linear_ratio = (confidence - THRESHOLD) / SCALE_RANGE
        
        sell_ratio = min(max(linear_ratio ** POWER, 0.0), 1.0)
        
        sold_shares = shares * sell_ratio
        cash += (sold_shares * price) * (1 - TAX_RATE)
        shares -= sold_shares
    
    '''
    if pred_class == 1 and confidence > 0.9:
        # 0.7~1.0 구간을 0.0~1.0 비율로 스케일링
        buy_ratio = (confidence - 0.9) / 0.1
        buy_ratio = min(max(buy_ratio, 0.0), 1.0) 
        # 현재 보유 현금에서 비율만큼 매수 (수수료 차감 후 주식 수 반영)
        invest_amount = cash * buy_ratio
        shares += (invest_amount * (1 - FEE_RATE)) / price
        cash -= invest_amount
        
    # [매도] AI가 '하락(0)'을 예측하고, 그 확신도가 0.9 초과일 때
    elif pred_class == 0 and confidence > 0.9:
        # 0.9~1.0 구간을 0.0~1.0 비율로 스케일링
        sell_ratio = (confidence - 0.9) / 0.1
        sell_ratio = min(max(sell_ratio, 0.0), 1.0)
        
        # 현재 보유 주식에서 비율만큼 매도 (세금 및 수수료 차감 후 현금 입금)
        sold_shares = shares * sell_ratio
        cash += (sold_shares * price) * (1 - TAX_RATE)
        shares -= sold_shares
    '''
    # [관망] 확신도가 0.5 ~ 0.7 사이면 아무 행동도 하지 않음 (Pass)

    # 5. 당일 장 마감 기준 총 자산 평가
    asset_value = cash + (shares * price)
    
    # 기록 저장
    dates.append(row['Date'])
    total_assets.append(asset_value)
    
    # 벤치마크: 첫날 1억원을 모두 코스피에 투자하고 계속 보유했을 때의 가치
    benchmark_value = (initial_capital / initial_price) * price
    benchmark_assets.append(benchmark_value)

# 6. 결과 시각화 (Matplotlib)
plt.figure(figsize=(12, 6))
plt.plot(dates, total_assets, label='Strategy Portfolio (AI)', color='blue')
plt.plot(dates, benchmark_assets, label='KOSPI Buy & Hold (Benchmark)', color='gray', alpha=0.7)

plt.title('Backtest Result: AI Strategy vs KOSPI (Including Fees/Taxes)')
plt.xlabel('Date')
plt.ylabel('Total Asset (KRW)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 그래프 출력
plt.show()

# 최종 결과 텍스트 출력
print(f"초기 자본: {initial_capital:,.0f} 원")
print(f"최종 자산: {total_assets[-1]:,.0f} 원")
print(f"전략 수익률: {(total_assets[-1] - initial_capital) / initial_capital * 100:.2f}%")
print(f"벤치마크 수익률: {(benchmark_assets[-1] - initial_capital) / initial_capital * 100:.2f}%")
