import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURATION
# =========================
START_DATE = "2023-11-01"
OUTPUT_DIR = "outputs/데이터 시각화"

MODELS = {
    "LSTM": {
        "path": "outputs/lstm_walk_forward_predictions.csv",
        "actual": "Actual_Target",
        "pred": "Predicted_Target"
    },
    "RandomForest": {
        "path": "outputs/rf_walk_forward_results.csv",
        "actual": "actual",
        "pred": "predicted"
    },
    "XGBoost": {
        "path": "outputs/xgb_results.csv",
        "actual": "Actual",
        "pred": "Predicted"
    },
    "LogisticRegression": {
        "path": "outputs/logistic_regression_walk_forward.csv",
        "actual": "Actual_Target",
        "pred": "Predicted_Target"
    }
}

def plot_cumulative_accuracy(name, config):
    if not os.path.exists(config["path"]):
        print(f"[{name}] File not found: {config['path']}")
        return

    df = pd.read_csv(config["path"])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= START_DATE].sort_values('Date')

    # Calculate Correctness (1 if hit, 0 if miss)
    df['Is_Correct'] = (df[config["actual"]] == df[config["pred"]]).astype(int)
    
    # Calculate Cumulative Accuracy (Running Mean in 0.0-1.0 scale)
    df['Cumulative_Accuracy'] = df['Is_Correct'].expanding().mean()

    # Final Stats
    final_acc = df['Cumulative_Accuracy'].iloc[-1]

    # Create Plot with Matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Cumulative_Accuracy'], color='skyblue', linewidth=2.5, label='Cumulative Accuracy')
    
    # Reference line at 0.5 (Random)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0.5 (Random)')

    plt.title(f"{name}: Cumulative Prediction Accuracy (From {START_DATE})\nFinal: {final_acc:.4f}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Accuracy (Ratio)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"cumulative_accuracy_{name.lower()}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[{name}] Cumulative accuracy chart saved to {save_path}")

def main():
    print("전체 모델(LSTM, RF, XGB, LR) 누적 정확도 그래프(Matplotlib) 생성을 시작합니다...")
    for name, config in MODELS.items():
        plot_cumulative_accuracy(name, config)
    print("\n그래프 생성이 모두 완료되었습니다.")

if __name__ == "__main__":
    main()
