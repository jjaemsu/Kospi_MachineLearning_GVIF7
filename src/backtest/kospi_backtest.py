import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# 1. 코스피 데이터 가져오기 (최근 10년)
kospi_data = yf.Ticker("^KS11").history(period="10y")
# 시계열 데이터를 맞추기 위해 시간대(timezone) 정보 제거
kospi_data.index = kospi_data.index.tz_localize(None) 

# 2. 백테스팅 결과 데이터 불러오기
df = pd.read_csv(r'C:\Users\이찬수\Desktop\gvif\outputs\walk_forward_predictions.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 3. 코스피 지수와 모델 예측 데이터 병합
merged_data = kospi_data[['Close']].join(df, how='inner')

# 4. 백테스팅 로직 적용 및 로그 출력
initial_capital = 1000000
cash = initial_capital
shares = 0
portfolio_values = []

buy_dates = []
buy_prices = []
sell_dates = []
sell_prices = []

current_state = 0 # 0: 현금, 1: 주식 보유

print(f"--- Backtesting Start (Initial Capital: {initial_capital:,.0f} KRW) ---")
log_lines = []

for date, row in merged_data.iterrows():
    price = row['Close']
    confidence = row['Probability_Up']
    predicted = row['Predicted_Target']

    if predicted == 1 and current_state == 0 and confidence >= 0.55:
        # 매수 로그
        shares = cash / price
        log = f"[{date.date()}] BUY  | Price: {price:,.2f} | Shares: {shares:.4f} | All-in"
        print(log)
        log_lines.append(log)
        cash = 0
        current_state = 1
        buy_dates.append(date)
        buy_prices.append(price)

    elif predicted == 0 and current_state == 1 and confidence < 0.45:
        # 매도 로그
        cash = shares * price
        log = f"[{date.date()}] SELL | Price: {price:,.2f} | Realized: {cash:,.0f} KRW"
        print(log)
        log_lines.append(log)
        shares = 0
        current_state = 0
        sell_dates.append(date)
        sell_prices.append(price)

    current_value = cash + (shares * price if shares > 0 else 0)
    portfolio_values.append(current_value)

merged_data['Portfolio_Value'] = portfolio_values
merged_data['Strategy_Return_Pct'] = (merged_data['Portfolio_Value'] / initial_capital - 1) * 100

# Calculate KOSPI Buy & Hold Cumulative Return for baseline comparison
kospi_initial = merged_data['Close'].iloc[0]
merged_data['KOSPI_Buy_Hold_Pct'] = (merged_data['Close'] / kospi_initial - 1) * 100

final_value = merged_data['Portfolio_Value'].iloc[-1]
kospi_final_return = merged_data['KOSPI_Buy_Hold_Pct'].iloc[-1]

end_log = f"\n--- Backtesting End ---\nFinal Asset Value: {final_value:,.2f} (Strategy Return: {(final_value/initial_capital - 1)*100:.2f}% | KOSPI B&H Return: {kospi_final_return:.2f}%)"
print(end_log)
log_lines.append(end_log)

# Save logs to file
with open('outputs/backtest_log.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

# 5. Plotly 누적 수익률 비교 차트 저장
fig = go.Figure()

# KOSPI 단순 보유 수익률 (Baseline)
fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data['KOSPI_Buy_Hold_Pct'], 
                         mode='lines', name='KOSPI Buy & Hold (%)', line=dict(color='gray', width=2, dash='dash')))

# LSTM 전략 수익률
fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data['Strategy_Return_Pct'], 
                         mode='lines', name='LSTM Strategy (%)', line=dict(color='blue', width=2.5)))

# 수익률이 0인 기준선 추가
fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

fig.update_layout(
    title='Cumulative Return Comparison: LSTM Strategy vs KOSPI Buy & Hold', 
    xaxis_title='Date', 
    yaxis_title='Cumulative Return (%)', 
    template='plotly_white',
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.write_html('outputs/cumulative_return_chart.html')
print("누적 수익률 비교 차트가 outputs/cumulative_return_chart.html 로 저장되었습니다.")