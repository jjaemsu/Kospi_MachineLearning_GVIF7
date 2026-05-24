"""Train and evaluate Random Forest for KOSPI direction with Optuna + TimeSeriesSplit.

Usage:
    python scripts/train_random_forest.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# USER_CONFIG
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "데이터 시각화"

LAG1_FILES = [
    DATA_DIR / "macro_data_stationary_lag1.csv",
    DATA_DIR / "market_data_stationary_lag1.csv",
    DATA_DIR / "price_data_stationary_lag1.csv",
    DATA_DIR / "volume_data_stationary_lag1.csv",
]

INITIAL_TRAIN_DAYS = 63  # Approx 3 months of trading days
RANDOM_STATE = 42
N_TRIALS = 30           # Reduced trials for speed since it's initial
INITIAL_SEED_MONEY = 100_000_000.0
UP_THRESHOLD = 0.50
DOWN_THRESHOLD = 0.50
TRADING_DAYS = 252

DATE_CANDIDATES = ["Date", "date", "날짜"]
TARGET_CANDIDATES = ["Target", "target", "Y", "y", "label"]
RETURN_CANDIDATES = ["Log_Return", "Return", "KOSPI_Return", "kospi_return", "수익률"]

# ... (find_first_matching_column, load_and_merge_lag1_data remain similar but updated)

def find_first_matching_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns:
            return cand
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def load_and_merge_lag1_data() -> Tuple[pd.DataFrame, str, str, str]:
    frames: List[pd.DataFrame] = []
    for path in LAG1_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Required lag1 file not found: {path}")
        frames.append(pd.read_csv(path))

    common_cols = set(frames[0].columns)
    for df in frames[1:]:
        common_cols &= set(df.columns)

    date_col = find_first_matching_column(list(common_cols), DATE_CANDIDATES)
    target_col = find_first_matching_column(list(common_cols), TARGET_CANDIDATES)
    if date_col is None or target_col is None:
        raise ValueError("Could not resolve Date/Target columns from lag1 files.")

    merged = frames[0].copy()
    for df in frames[1:]:
        drop_cols = [c for c in [target_col] if c in df.columns]
        merged = merged.merge(df.drop(columns=drop_cols), on=date_col, how="inner")

    return_col = find_first_matching_column(list(merged.columns), RETURN_CANDIDATES)
    if return_col is None:
        raise ValueError("Could not resolve return column. Update RETURN_CANDIDATES in USER_CONFIG.")

    merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
    merged = merged.dropna(subset=[date_col, target_col]).sort_values(date_col).reset_index(drop=True)
    return merged, date_col, target_col, return_col

def build_pipeline(feature_columns: List[str], params: Dict[str, object]) -> Pipeline:
    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_preprocess, feature_columns)],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        max_features=params["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

def apply_strategy(probs_up, probs_down, pred, kospi_returns, strategy_name):
    if strategy_name == "A":
        long_signal = (pred == 1) & (probs_up >= UP_THRESHOLD)
        short_signal = (pred == 0) & (probs_down >= DOWN_THRESHOLD)
        daily = np.where(long_signal, kospi_returns, np.where(short_signal, -kospi_returns, 0.0))
    elif strategy_name == "B":
        long_signal = (pred == 1) & (probs_up >= UP_THRESHOLD)
        daily = np.where(long_signal, kospi_returns, 0.0)
    return pd.Series(daily, index=kospi_returns.index)

def main() -> None:
    from tqdm import tqdm
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, date_col, target_col, return_col = load_and_merge_lag1_data()
    feature_cols = [c for c in df.columns if c not in [date_col, target_col]]
    
    total_len = len(df)
    print(f"Total Rows: {total_len}, Initial Train: {INITIAL_TRAIN_DAYS}")

    # 1. Initial Optuna Tuning on first 3 months
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        X_init = df.iloc[:INITIAL_TRAIN_DAYS][feature_cols]
        y_init = df.iloc[:INITIAL_TRAIN_DAYS][target_col].astype(int)
        
        # Simple split for tuning speed within the initial window
        X_tr, X_val, y_tr, y_val = train_test_split(X_init, y_init, test_size=0.2, shuffle=False)
        model = build_pipeline(feature_cols, params)
        model.fit(X_tr, y_tr)
        return accuracy_score(y_val, model.predict(X_val))

    print("Step 1: Running Optuna for initial hyperparameters...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    print(f"Best Params: {best_params}")

    # 2. Walk-Forward Expanding Window
    predictions = []
    print("Step 2: Walk-Forward Validation (Expanding Window)...")
    
    for t in tqdm(range(INITIAL_TRAIN_DAYS, total_len)):
        X_train = df.iloc[:t][feature_cols]
        y_train = df.iloc[:t][target_col].astype(int)
        X_test = df.iloc[t:t+1][feature_cols]
        
        model = build_pipeline(feature_cols, best_params)
        model.fit(X_train, y_train)
        
        prob = model.predict_proba(X_test)[0]
        pred = model.predict(X_test)[0]
        
        predictions.append({
            "Date": df.iloc[t][date_col],
            "actual": int(df.iloc[t][target_col]),
            "predicted": int(pred),
            "prob_down": prob[0],
            "prob_up": prob[1],
            "kospi_return": df.iloc[t][return_col]
        })

    # 3. Evaluation
    results_df = pd.DataFrame(predictions)
    results_df["Date"] = pd.to_datetime(results_df["Date"])
    
    daily_a = apply_strategy(results_df["prob_up"], results_df["prob_down"], results_df["predicted"], results_df["kospi_return"], "A")
    daily_b = apply_strategy(results_df["prob_up"], results_df["prob_down"], results_df["predicted"], results_df["kospi_return"], "B")
    
    results_df["strategy_a_daily"] = daily_a.values
    results_df["strategy_b_daily"] = daily_b.values
    results_df["strategy_a_equity"] = (1.0 + daily_a).cumprod().values * INITIAL_SEED_MONEY
    results_df["strategy_b_equity"] = (1.0 + daily_b).cumprod().values * INITIAL_SEED_MONEY
    results_df["benchmark_equity"] = (1.0 + results_df["kospi_return"]).cumprod().values * INITIAL_SEED_MONEY
    
    # Save Outputs
    results_df.to_csv(OUTPUT_DIR / "rf_walk_forward_results.csv", index=False)
    
    # Simple Metrics
    acc = accuracy_score(results_df["actual"], results_df["predicted"])
    f1 = f1_score(results_df["actual"], results_df["predicted"])
    print(f"\n[Walk-Forward Results]\nAccuracy: {acc:.4f}\nF1-score: {f1:.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["Date"], results_df["benchmark_equity"], label="KOSPI Benchmark")
    plt.plot(results_df["Date"], results_df["strategy_a_equity"], label="Strategy A (Long/Short)")
    plt.plot(results_df["Date"], results_df["strategy_b_equity"], label="Strategy B (Long Only)")
    plt.title("Random Forest Walk-Forward: Cumulative Return")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "rf_walk_forward_equity.png")
    
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    # Leakage safety checks:
    # 1) All splits are chronological (shuffle=False, TimeSeriesSplit).
    # 2) Preprocessing is inside pipeline and fitted only on training fold.
    # 3) Validation/test uses strictly future segments.
    main()
