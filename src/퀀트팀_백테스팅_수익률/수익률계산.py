import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. 데이터 불러오기
df_data = pd.read_csv("kospi_data.csv")
df_pred = pd.read_csv("kospi_final_daily_results_1차.csv") 

# 2. 날짜 데이터 전처리
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_pred['Date'] = pd.to_datetime(df_pred['Date'])

# 3. 데이터 병합
df = pd.merge(df_pred[['Date', 'Predicted']], 
              df_data[['Date', 'Close', 'Return']], 
              on='Date', how='inner')
df = df.sort_values('Date').reset_index(drop=True)

# =====================================================================
# [수정 핵심] 미래참조(Look-ahead Bias) 방지 로직 추가
# t일(어제) 장 마감에 예측한 시그널(Predicted)을 t+1일(오늘)의 수익률(Return)에 적용.
# =====================================================================
df['Signal'] = df['Predicted'].shift(1)

# 첫째 날은 어제의 시그널이 없으므로(NaN) 보수적으로 0(현금관망) 처리합니다.
df['Signal'] = df['Signal'].fillna(0)
# =====================================================================

# 4. 전략 수익률 계산
# [Long-Only] '당일 적용 시그널(Signal)'이 1이면 코스피 수익률 취함, 0이면 0 (현금보유)
df['Long_Only_Return'] = np.where(df['Signal'] == 1, df['Return'], 0)

# [Long-Short] '당일 적용 시그널(Signal)'이 1이면 코스피 수익률 취함, 0이면 코스피 수익률의 반대(-Return)를 취함
df['Long_Short_Return'] = np.where(df['Signal'] == 1, df['Return'], -df['Return'])

# 누적 수익률 계산 (기본값 1.0 기준)
df['KOSPI_Cum_Return'] = (1 + df['Return']).cumprod()
df['Long_Only_Cum_Return'] = (1 + df['Long_Only_Return']).cumprod()
df['Long_Short_Cum_Return'] = (1 + df['Long_Short_Return']).cumprod()

# =====================================================================
# 매매 일지(Trading Log) 생성 및 CSV 추출 (수정됨)
# =====================================================================
# 전날 대비 포지션 변화를 확인합니다. (Predicted가 아닌 당일 'Signal' 기준)
df['Position_Change'] = df['Signal'].diff()

# 백테스트 첫날의 매매 액션 처리 (이전 데이터가 없으므로 NaN 처리 방지)
if df.loc[0, 'Signal'] == 1:
    df.loc[0, 'Position_Change'] = 1
else:
    df.loc[0, 'Position_Change'] = 0

# 롱온리(Long-Only) 기준 매매 액션을 텍스트로 변환하는 함수
def get_trade_action(row):
    if row['Position_Change'] == 1:
        return "신규 매수 (Buy)"
    elif row['Position_Change'] == -1:
        return "전량 매도 (Sell)"
    elif row['Position_Change'] == 0 and row['Signal'] == 1:
        return "매수 보유 (Hold)"
    else:
        return "현금 관망 (Cash)"

df['Trade_Action'] = df.apply(get_trade_action, axis=1)

# CSV로 뽑아낼 컬럼들만 정리 (Signal 컬럼 추가)
log_df = df[['Date', 'Close', 'Predicted', 'Signal', 'Trade_Action', 
             'Return', 'Long_Only_Return', 'Long_Only_Cum_Return',
             'Long_Short_Return', 'Long_Short_Cum_Return']].copy()

log_df.columns = ['날짜', '코스피_종가', '전일_모델_예측(내일방향)', '당일_적용_시그널', '당일_매매_액션(Long-Only)', 
                  '당일_코스피_수익률', '당일_전략_수익률(롱온리)', '누적_수익률(롱온리)',
                  '당일_전략_수익률(롱숏)', '누적_수익률(롱숏)']

# 엑셀에서 한글이 깨지지 않도록 encoding='utf-8-sig' 적용하여 저장
log_df.to_csv("trading_log.csv", index=False, encoding='utf-8-sig')
print("✅ [저장 완료] 매일의 매매 내역과 수익률이 담긴 'trading_log.csv' 파일이 생성되었습니다.\n")
# =====================================================================

# 5. 성과 지표(Metrics) 계산 함수
def calculate_metrics(returns):
    cum_return = (1 + returns).cumprod() - 1
    total_return = cum_return.iloc[-1]
    
    roll_max = (1 + returns).cumprod().cummax()
    drawdown = (1 + returns).cumprod() / roll_max - 1.0
    mdd = drawdown.min()
    
    if returns.std() != 0:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe = 0
        
    return total_return, mdd, sharpe

kospi_ret, kospi_mdd, kospi_sharpe = calculate_metrics(df['Return'])
lo_ret, lo_mdd, lo_sharpe = calculate_metrics(df['Long_Only_Return'])
ls_ret, ls_mdd, ls_sharpe = calculate_metrics(df['Long_Short_Return'])

# 6. 결과 출력
print(f"[KOSPI 단순 보유] 누적 수익률: {kospi_ret*100:.2f}%, MDD: {kospi_mdd*100:.2f}%, 샤프 지수: {kospi_sharpe:.2f}")
print(f"[Long-Only 전략] 누적 수익률: {lo_ret*100:.2f}%, MDD: {lo_mdd*100:.2f}%, 샤프 지수: {lo_sharpe:.2f}")
print(f"[Long-Short 전략] 누적 수익률: {ls_ret*100:.2f}%, MDD: {ls_mdd*100:.2f}%, 샤프 지수: {ls_sharpe:.2f}")

# 7. 시각화 (그래프 그리기)
plt.figure(figsize=(12, 6))

plt.plot(df['Date'], df['KOSPI_Cum_Return'], 
         label=f'KOSPI ({kospi_ret*100:.1f}%)', color='gray', alpha=0.7)
plt.plot(df['Date'], df['Long_Only_Cum_Return'], 
         label=f'Long-Only ({lo_ret*100:.1f}%)', color='dodgerblue')
plt.plot(df['Date'], df['Long_Short_Cum_Return'], 
         label=f'Long-Short ({ls_ret*100:.1f}%)', color='crimson')

plt.title('KOSPI vs Long-Only vs Long-Short Backtest', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Cumulative Return (Base = 1.0)', fontsize=11)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()

# 이미지 저장 및 출력
plt.savefig('backtest_ls_result.png')
plt.show()