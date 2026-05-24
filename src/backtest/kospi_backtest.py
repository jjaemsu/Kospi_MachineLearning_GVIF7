import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# CONFIGURATION
# =========================
BACKTEST_START_DATE = "2023-11-01"
INITIAL_CAPITAL = 1_000_000
TRADING_DAYS = 252

# 백테스팅을 수행할 모델과 해당 예측 결과 파일 경로 매핑
MODEL_CONFIGS = {
    "LSTM": {"path": "outputs/walk_forward_predictions.csv", "buy_thr": 0.55, "sell_thr": 0.45},
    "Random_Forest": {"path": "outputs/rf_walk_forward_results.csv", "buy_thr": 0.55, "sell_thr": 0.45},
    "XGBoost": {"path": "outputs/xgb_results.csv", "buy_thr": 0.50, "sell_thr": 0.50},
    "Logistic_Regression": {"path": "outputs/logistic_regression_walk_forward.csv", "buy_thr": 0.52, "sell_thr": 0.48},
    "LightGBM": {"path": "outputs/lgbm_walk_forward_results.csv", "buy_thr": 0.55, "sell_thr": 0.45}
}

def calculate_mdd(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()

def calculate_sharpe(daily_returns):
    if daily_returns.std() == 0:
        return 0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS)

def run_backtest_for_model(model_name, config):
    input_csv = config["path"]
    buy_thr = config["buy_thr"]
    sell_thr = config["sell_thr"]
    
    # 1. 코스피 데이터 가져오기
    try:
        # 넉넉하게 가져와서 필터링
        kospi_data = yf.Ticker("^KS11").history(period="10y")
        kospi_data.index = kospi_data.index.tz_localize(None) 
    except Exception as e:
        print(f"[{model_name}] KOSPI 데이터 다운로드 실패: {e}")
        return None

    # 2. 백테스팅 결과 데이터 불러오기
    if not os.path.exists(input_csv):
        print(f"[{model_name}] 예측 파일이 없습니다. (건너뜀): {input_csv}")
        return None

    df = pd.read_csv(input_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 컬럼명 매핑 (통합 처리)
    mapping = {
        'prob_up': 'Probability_Up',
        'Probability_Up': 'Probability_Up',
        'Probability_Target': 'Probability_Up',
        'Confidence': 'Confidence',
        'predicted': 'Predicted_Target',
        'Predicted_Target': 'Predicted_Target',
        'Predicted': 'Predicted_Target',
        'Actual_Target': 'Actual_Target',
        'Actual': 'Actual_Target',
        'actual': 'Actual_Target'
    }
    for old_col, new_col in mapping.items():
        if old_col in df.columns and old_col != new_col:
            df[new_col] = df[old_col]

    # Confidence 보정
    if 'Confidence' in df.columns and 'Probability_Up' not in df.columns:
        df['Probability_Up'] = df.apply(lambda x: x['Confidence'] if x['Predicted_Target'] == 1 else 1 - x['Confidence'], axis=1)

    # 3. 기간 필터링 및 병합 (2023-09-01 기준)
    merged_data = kospi_data[['Close']].join(df, how='inner')
    merged_data = merged_data[merged_data.index >= BACKTEST_START_DATE]

    if merged_data.empty:
        print(f"[{model_name}] 지정된 시작일({BACKTEST_START_DATE}) 이후 데이터가 없습니다.")
        return None

    print(f"\n========== [{model_name}] 백테스트 시작 (기간: {merged_data.index[0].date()} ~ {merged_data.index[-1].date()}) ==========")

    # 4. 백테스팅 로직
    cash = INITIAL_CAPITAL
    shares = 0
    portfolio_values = []
    current_state = 0 # 0: 현금, 1: 주식
    
    buy_signals = []
    sell_signals = []
    trade_returns = []
    last_buy_price = 0

    for date, row in merged_data.iterrows():
        price = row['Close']
        prob_up = row['Probability_Up']
        predicted = row['Predicted_Target']

        is_buy = False
        is_sell = False

        if model_name == "XGBoost":
            confidence = prob_up if predicted == 1 else (1 - prob_up)
            if confidence >= buy_thr:
                if predicted == 1 and current_state == 0: is_buy = True
                elif predicted == 0 and current_state == 1: is_sell = True
        else:
            if predicted == 1 and current_state == 0 and prob_up >= buy_thr: is_buy = True
            elif predicted == 0 and current_state == 1 and prob_up < sell_thr: is_sell = True

        if is_buy:
            shares = cash / price
            cash = 0
            current_state = 1
            buy_signals.append(date)
            last_buy_price = price
        elif is_sell:
            cash = shares * price
            trade_returns.append(cash / (shares * last_buy_price) - 1)
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
    # ML 지표
    y_true = merged_data['Actual_Target'].astype(int)
    y_pred = merged_data['Predicted_Target'].astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 재무 지표
    final_return = merged_data['Strategy_Return_Pct'].iloc[-1]
    kospi_final_return = merged_data['KOSPI_Return_Pct'].iloc[-1]
    mdd = calculate_mdd(merged_data['Portfolio_Value']) * 100
    daily_rets = merged_data['Portfolio_Value'].pct_change().fillna(0)
    sharpe = calculate_sharpe(daily_rets)
    total_trades = len(buy_signals)

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Strategy_Return(%)": final_return,
        "KOSPI_Return(%)": kospi_final_return,
        "MDD(%)": mdd,
        "Sharpe_Ratio": sharpe,
        "Total_Trades": total_trades
    }

    # 6. 시각화 (1): 누적 수익률 차트
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(x=merged_data.index, y=merged_data['KOSPI_Return_Pct'], 
                                 mode='lines', name='KOSPI (Benchmark) %', line=dict(color='lightgray', dash='dash')))
    fig_ret.add_trace(go.Scatter(x=merged_data.index, y=merged_data['Strategy_Return_Pct'], 
                                 mode='lines', name=f'{model_name} Strategy %', line=dict(color='blue', width=2)))
    
    if buy_signals:
        fig_ret.add_trace(go.Scatter(x=buy_signals, y=[merged_data.loc[d, 'Strategy_Return_Pct'] for d in buy_signals],
                                     mode='markers', name='Buy', marker=dict(color='red', size=10, symbol='triangle-up'), hoverinfo='skip'))
    if sell_signals:
        fig_ret.add_trace(go.Scatter(x=sell_signals, y=[merged_data.loc[d, 'Strategy_Return_Pct'] for d in sell_signals],
                                     mode='markers', name='Sell', marker=dict(color='blue', size=10, symbol='triangle-down'), hoverinfo='skip'))

    fig_ret.update_layout(
        title=f"<b>{model_name} Backtest: Cumulative Return</b> (From {BACKTEST_START_DATE})<br>" +
              f"<span style='font-size:12px;'>Return {final_return:.2f}%, MDD {mdd:.2f}%, Sharpe {sharpe:.2f}</span>",
        xaxis_title='Date', yaxis_title='Cumulative Return (%)', template='plotly_white', hovermode="x unified"
    )

    # 7. 시각화 (2): 누적 예측 정확도 차트 (Cumulative Accuracy)
    # 실제값과 예측값이 일치하는지 확인 (1: 맞음, 0: 틀림)
    merged_data['Is_Correct'] = (merged_data['Actual_Target'] == merged_data['Predicted_Target']).astype(int)
    # 누적 정확도 계산 (누적 맞춘 개수 / 진행된 날짜 수)
    merged_data['Cumulative_Accuracy'] = merged_data['Is_Correct'].expanding().mean() * 100

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=merged_data.index, y=merged_data['Cumulative_Accuracy'],
                                 mode='lines', name='Cumulative Accuracy (%)', line=dict(color='green', width=2)))
    
    # 50% 기준선 (무작위 예측 기준)
    fig_acc.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50% (Random)")

    fig_acc.update_layout(
        title=f"<b>{model_name}: Cumulative Prediction Accuracy</b><br>" +
              f"<span style='font-size:12px;'>Final Accuracy: {acc:.2%} | F1-Score: {f1:.2f}</span>",
        xaxis_title='Date', yaxis_title='Accuracy (%)', template='plotly_white',
        yaxis=dict(range=[min(40, merged_data['Cumulative_Accuracy'].min()), max(60, merged_data['Cumulative_Accuracy'].max())])
    )

    output_dir = "outputs/데이터 시각화"
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 저장
    fig_ret.write_html(os.path.join(output_dir, f"backtest_chart_{model_name.lower()}.html"))
    fig_acc.write_html(os.path.join(output_dir, f"accuracy_chart_{model_name.lower()}.html"))
    
    print(f"[{model_name}] 리포트 및 정확도 차트 저장 완료.")
    
    return metrics

def main():
    print(f"백테스트 기간 통일 ({BACKTEST_START_DATE} ~ ) 및 지표 산출을 시작합니다...")
    all_metrics = []
    for model_name, config in MODEL_CONFIGS.items():
        m = run_backtest_for_model(model_name, config)
        if m: all_metrics.append(m)
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = "outputs/데이터 시각화/models_summary_report.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n전체 모델 비교 리포트 저장 완료: {summary_path}")
        print(summary_df.drop(columns=['Accuracy', 'Precision', 'Recall', 'F1-Score']).to_string(index=False))

if __name__ == "__main__":
    main()
