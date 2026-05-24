import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score, f1_score

# =========================
# CONFIGURATION
# =========================
BACKTEST_START_DATE = "2023-11-01"
INITIAL_CAPITAL = 1_000_000
TRADING_DAYS = 252
DATA_DIR = "data/xgboost_data"
OUTPUT_DIR = "outputs/xgboost_batch_backtest"

def calculate_mdd(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()

def calculate_sharpe(daily_returns):
    if daily_returns.std() == 0:
        return 0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS)

def run_backtest_for_file(file_name):
    input_csv = os.path.join(DATA_DIR, file_name)
    model_name = file_name.replace(".csv", "")
    
    # 1. 코스피 데이터 가져오기
    try:
        kospi_data = yf.Ticker("^KS11").history(period="10y")
        kospi_data.index = kospi_data.index.tz_localize(None) 
    except Exception as e:
        print(f"[{model_name}] KOSPI 데이터 다운로드 실패: {e}")
        return None

    # 2. 데이터 불러오기 (헤더 있음 확인됨)
    df = pd.read_csv(input_csv)
    
    # 컬럼명 매핑 (XGBoost 결과 파일 포맷에 맞춤)
    mapping = {
        'Actual': 'Actual_Target',
        'Predicted': 'Predicted_Target',
        'Confidence': 'Confidence'
    }
    df = df.rename(columns=mapping)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Probability_Up 계산 (사용자 요청: 0.5 이상일 때 방향성 수용)
    df['Probability_Up'] = df.apply(lambda x: x['Confidence'] if x['Predicted_Target'] == 1 else 1 - x['Confidence'], axis=1)

    # 3. 기간 필터링 및 병합
    merged_data = kospi_data[['Close']].join(df, how='inner')
    merged_data = merged_data[merged_data.index >= BACKTEST_START_DATE]

    if merged_data.empty:
        print(f"[{model_name}] 데이터가 없습니다.")
        return None

    # 4. 백테스팅 로직 (XGBoost 맞춤: Confidence >= 0.5)
    cash = INITIAL_CAPITAL
    shares = 0
    portfolio_values = []
    current_state = 0 
    buy_signals = []
    sell_signals = []

    for date, row in merged_data.iterrows():
        price = row['Close']
        prob_up = row['Probability_Up']
        predicted = row['Predicted_Target']

        # XGBoost 로직: Confidence >= 0.5 이면 1일 때 매수, 0일 때 매도
        confidence_val = prob_up if predicted == 1 else (1 - prob_up)
        
        is_buy = (predicted == 1 and current_state == 0 and confidence_val >= 0.50)
        is_sell = (predicted == 0 and current_state == 1 and confidence_val >= 0.50)

        if is_buy:
            shares = cash / price
            cash = 0
            current_state = 1
            buy_signals.append(date)
        elif is_sell:
            cash = shares * price
            shares = 0
            current_state = 0
            sell_signals.append(date)

        current_value = cash + (shares * price if shares > 0 else 0)
        portfolio_values.append(current_value)

    merged_data['Portfolio_Value'] = portfolio_values
    merged_data['Strategy_Return_Pct'] = (merged_data['Portfolio_Value'] / INITIAL_CAPITAL - 1) * 100
    
    kospi_initial = merged_data['Close'].iloc[0]
    merged_data['KOSPI_Return_Pct'] = (merged_data['Close'] / kospi_initial - 1) * 100

    # 5. 지표 계산
    y_true = merged_data['Actual_Target'].astype(int)
    y_pred = merged_data['Predicted_Target'].astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    final_return = merged_data['Strategy_Return_Pct'].iloc[-1]
    mdd = calculate_mdd(merged_data['Portfolio_Value']) * 100
    sharpe = calculate_sharpe(merged_data['Portfolio_Value'].pct_change().fillna(0))

    metrics = {
        "File": file_name,
        "Accuracy": acc,
        "F1-Score": f1,
        "Return(%)": final_return,
        "MDD(%)": mdd,
        "Sharpe": sharpe,
        "Trades": len(buy_signals)
    }

    # 6. 시각화
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data['KOSPI_Return_Pct'], name='KOSPI %', line=dict(color='lightgray', dash='dash')))
    fig.add_trace(go.Scatter(x=merged_data.index, y=merged_data['Strategy_Return_Pct'], name='Strategy %', line=dict(color='blue', width=2)))
    
    if buy_signals:
        fig.add_trace(go.Scatter(x=buy_signals, y=[merged_data.loc[d, 'Strategy_Return_Pct'] for d in buy_signals],
                                 mode='markers', name='Buy', marker=dict(color='red', size=8, symbol='triangle-up'), hoverinfo='skip'))
    if sell_signals:
        fig.add_trace(go.Scatter(x=sell_signals, y=[merged_data.loc[d, 'Strategy_Return_Pct'] for d in sell_signals],
                                 mode='markers', name='Sell', marker=dict(color='blue', size=8, symbol='triangle-down'), hoverinfo='skip'))

    fig.update_layout(
        title=f"<b>XGBoost Batch: {model_name}</b><br>Return: {final_return:.2f}%, MDD: {mdd:.2f}%, Acc: {acc:.2%}",
        xaxis_title='Date', yaxis_title='Return (%)', template='plotly_white'
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.write_html(os.path.join(OUTPUT_DIR, f"chart_{model_name}.html"))
    return metrics

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found.")
        return
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"총 {len(files)}개 파일에 대한 배치 백테스팅을 시작합니다...")
    
    results = []
    for f in files:
        m = run_backtest_for_file(f)
        if m: 
            results.append(m)
            print(f"- {f} 완료 (수익률: {m['Return(%)']:.2f}%)")
    
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(OUTPUT_DIR, "batch_summary.csv"), index=False, encoding="utf-8-sig")
        print(f"\n배치 결과 저장 완료: {OUTPUT_DIR}/batch_summary.csv")
        print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
