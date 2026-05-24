import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

LAG1_FILES = [
    DATA_DIR / "macro_data_stationary_lag1.csv",
    DATA_DIR / "market_data_stationary_lag1.csv",
    DATA_DIR / "price_data_stationary_lag1.csv",
    DATA_DIR / "volume_data_stationary_lag1.csv",
]

INITIAL_TRAIN_DAYS = 63
RANDOM_STATE = 42

def load_and_merge_data():
    frames = []
    for path in LAG1_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        frames.append(pd.read_csv(path))

    # Merge on Date
    df = frames[0]
    for next_df in frames[1:]:
        drop_cols = [c for c in ['Target'] if c in next_df.columns]
        df = df.merge(next_df.drop(columns=drop_cols), on='Date', how='inner')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading and merging lag1 datasets...")
    df = load_and_merge_data()
    
    feature_cols = [c for c in df.columns if c not in ['Date', 'Target', 'Log_Return', 'Return', 'KOSPI_Return']]
    # Ensure all selected features are numeric
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'Target'
    
    total_len = len(df)
    print(f"Total Rows: {total_len}, Initial Train Window: {INITIAL_TRAIN_DAYS}")
    print(f"Number of Features: {len(feature_cols)}")

    predictions = []
    
    print("Starting Walk-Forward Validation (Logistic Regression)...")
    for t in tqdm(range(INITIAL_TRAIN_DAYS, total_len)):
        # Data split
        X_train = df.iloc[:t][feature_cols]
        y_train = df.iloc[:t][target_col].astype(int)
        X_test = df.iloc[t:t+1][feature_cols]
        y_actual = df.iloc[t][target_col]
        
        # Preprocessing: Logistic Regression requires scaling
        # Handle potential NaNs just in case
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model Fit (liblinear is good for small datasets)
        model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, solver='liblinear', max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        prob = model.predict_proba(X_test_scaled)[0]
        pred = model.predict(X_test_scaled)[0]
        
        predictions.append({
            "Date": df.iloc[t]['Date'],
            "Actual_Target": int(y_actual),
            "Predicted_Target": int(pred),
            "Probability_Up": prob[1]
        })

    # Results
    results_df = pd.DataFrame(predictions)
    acc = accuracy_score(results_df['Actual_Target'], results_df['Predicted_Target'])
    f1 = f1_score(results_df['Actual_Target'], results_df['Predicted_Target'])
    
    print(f"\n[Logistic Regression Results]\nAccuracy: {acc:.4f}\nF1-score: {f1:.4f}")
    
    output_path = OUTPUT_DIR / "logistic_regression_walk_forward.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()
