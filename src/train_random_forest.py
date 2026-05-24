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
OUTPUT_DIR = ROOT_DIR / "outputs"

LAG1_FILES = [
    DATA_DIR / "macro_data_stationary_lag1.csv",
    DATA_DIR / "market_data_stationary_lag1.csv",
    DATA_DIR / "price_data_stationary_lag1.csv",
    DATA_DIR / "volume_data_stationary_lag1.csv",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_TRIALS = 50
N_SPLITS = 5
INITIAL_SEED_MONEY = 100_000_000.0
UP_THRESHOLD = 0.60
DOWN_THRESHOLD = 0.60
TRADING_DAYS = 252

DATE_CANDIDATES = ["Date", "date", "날짜"]
TARGET_CANDIDATES = ["Target", "target", "Y", "y", "label"]
RETURN_CANDIDATES = ["Log_Return", "Return", "KOSPI_Return", "kospi_return", "수익률"]


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
            # RF does not require scaling, but keep preprocessing explicit for cross-model comparability.
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


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(daily_returns: pd.Series) -> float:
    std = daily_returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((daily_returns.mean() / std) * np.sqrt(TRADING_DAYS))


def apply_strategy(
    probs_up: np.ndarray,
    probs_down: np.ndarray,
    predicted: np.ndarray,
    kospi_returns: pd.Series,
    strategy_name: str,
) -> Tuple[pd.Series, pd.Series]:
    if strategy_name == "A":
        long_signal = (predicted == 1) & (probs_up >= UP_THRESHOLD)
        short_signal = (predicted == 0) & (probs_down >= DOWN_THRESHOLD)
        daily = np.where(long_signal, kospi_returns.values, np.where(short_signal, -kospi_returns.values, 0.0))
    elif strategy_name == "B":
        long_signal = (predicted == 1) & (probs_up >= UP_THRESHOLD)
        daily = np.where(long_signal, kospi_returns.values, 0.0)
    else:
        raise ValueError("strategy_name must be 'A' or 'B'.")

    daily_ret = pd.Series(daily, index=kospi_returns.index)
    equity = (1.0 + daily_ret).cumprod() * INITIAL_SEED_MONEY
    return daily_ret, equity


def compute_strategy_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    daily_returns: pd.Series,
    equity_curve: pd.Series,
) -> Dict[str, float]:
    cumulative_return = float(equity_curve.iloc[-1] / INITIAL_SEED_MONEY - 1.0)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown(equity_curve),
    }


def objective_factory(X_train: pd.DataFrame, y_train: pd.Series, r_train: pd.Series, feature_cols: List[str]):
    splitter = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

        fold_sharpe: List[float] = []
        fold_cumret: List[float] = []
        fold_acc: List[float] = []

        for tr_idx, val_idx in splitter.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            r_val = r_train.iloc[val_idx]

            model = build_pipeline(feature_cols, params)
            model.fit(X_tr, y_tr)

            pred = model.predict(X_val)
            proba = model.predict_proba(X_val)
            p_down = proba[:, 0]
            p_up = proba[:, 1]

            # Objective evaluates financial performance using Strategy A (long/short/cash).
            daily_ret, equity = apply_strategy(p_up, p_down, pred, r_val, strategy_name="A")
            metrics = compute_strategy_metrics(y_val, pred, daily_ret, equity)

            fold_sharpe.append(metrics["sharpe_ratio"])
            fold_cumret.append(metrics["cumulative_return"])
            fold_acc.append(metrics["accuracy"])

        mean_sharpe = float(np.mean(fold_sharpe))
        mean_cumret = float(np.mean(fold_cumret))
        mean_acc = float(np.mean(fold_acc))

        # Lexicographic priority encoding: Sharpe > cumulative return > accuracy.
        score = mean_sharpe * 1_000_000 + mean_cumret * 1_000 + mean_acc

        trial.set_user_attr("mean_sharpe", mean_sharpe)
        trial.set_user_attr("mean_cumret", mean_cumret)
        trial.set_user_attr("mean_accuracy", mean_acc)
        return score

    return objective


def run_optuna(X_train: pd.DataFrame, y_train: pd.Series, r_train: pd.Series, feature_cols: List[str]) -> optuna.Study:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective_factory(X_train, y_train, r_train, feature_cols), n_trials=N_TRIALS)
    return study


def evaluate_on_test(
    model: Pipeline,
    test_frame: pd.DataFrame,
    date_col: str,
    target_col: str,
    return_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    X_test = test_frame.drop(columns=[date_col, target_col])
    y_test = test_frame[target_col].astype(int)
    r_test = test_frame[return_col].astype(float)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    p_down = proba[:, 0]
    p_up = proba[:, 1]

    benchmark_equity = (1.0 + r_test).cumprod() * INITIAL_SEED_MONEY
    daily_a, equity_a = apply_strategy(p_up, p_down, pred, r_test, strategy_name="A")
    daily_b, equity_b = apply_strategy(p_up, p_down, pred, r_test, strategy_name="B")

    result_df = pd.DataFrame(
        {
            date_col: test_frame[date_col].values,
            "actual": y_test.values,
            "predicted": pred,
            "prob_down": p_down,
            "prob_up": p_up,
            "kospi_return": r_test.values,
            "benchmark_cumulative_return": benchmark_equity.values,
            "strategy_a_cumulative_return": equity_a.values,
            "strategy_b_cumulative_return": equity_b.values,
            "strategy_a_daily_return": daily_a.values,
            "strategy_b_daily_return": daily_b.values,
        }
    )

    correct = (result_df["actual"] == result_df["predicted"]).astype(int)
    result_df["cumulative_accuracy"] = correct.cumsum() / np.arange(1, len(result_df) + 1)

    comparison = pd.DataFrame(
        [
            {
                "strategy": "A_long_short_cash",
                **compute_strategy_metrics(y_test, pred, daily_a, equity_a),
            },
            {
                "strategy": "B_long_cash_only",
                **compute_strategy_metrics(y_test, pred, daily_b, equity_b),
            },
        ]
    )

    return result_df, comparison, confusion_matrix(y_test, pred)


def save_optuna_artifacts(study: optuna.Study) -> None:
    history = pd.DataFrame(
        {
            "trial": [t.number for t in study.trials],
            "score": [t.value for t in study.trials],
            "mean_sharpe": [t.user_attrs.get("mean_sharpe", np.nan) for t in study.trials],
            "mean_cumulative_return": [t.user_attrs.get("mean_cumret", np.nan) for t in study.trials],
            "mean_accuracy": [t.user_attrs.get("mean_accuracy", np.nan) for t in study.trials],
        }
    )

    plt.figure(figsize=(12, 6))
    plt.plot(history["trial"], history["score"], label="Trial Score", alpha=0.6)
    plt.plot(history["trial"], history["score"].cummax(), label="Best Score So Far", linewidth=2)
    plt.title("Optuna Optimization History: Random Forest")
    plt.xlabel("Trial")
    plt.ylabel("Objective Score")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_optuna_history.png", dpi=150)
    plt.close()


def plot_cumulative_returns(result_df: pd.DataFrame, date_col: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(result_df[date_col], result_df["benchmark_cumulative_return"], label="KOSPI Benchmark")
    plt.plot(result_df[date_col], result_df["strategy_a_cumulative_return"], label="Strategy A: Long/Short/Cash")
    plt.plot(result_df[date_col], result_df["strategy_b_cumulative_return"], label="Strategy B: Long/Cash")
    plt.title("Cumulative Return: KOSPI vs Random Forest Strategies")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (KRW)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_cumulative_return_comparison.png", dpi=150)
    plt.close()


def plot_cumulative_accuracy(result_df: pd.DataFrame, date_col: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(result_df[date_col], result_df["cumulative_accuracy"], label="Cumulative Accuracy")
    plt.title("Cumulative Prediction Accuracy: Random Forest")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rf_cumulative_prediction_accuracy.png", dpi=150)
    plt.close()


def save_feature_importance(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    importances = model.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_df.to_csv(OUTPUT_DIR / "rf_feature_importance.csv", index=False)
    return fi_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, date_col, target_col, return_col = load_and_merge_lag1_data()
    feature_cols = [c for c in df.columns if c not in [date_col, target_col]]

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, shuffle=False)
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    r_train = train_df[return_col].astype(float)

    study = run_optuna(X_train, y_train, r_train, feature_cols)
    best_params = study.best_params

    best_model = build_pipeline(feature_cols, best_params)
    best_model.fit(X_train, y_train)

    results_df, comparison_df, cm = evaluate_on_test(best_model, test_df, date_col, target_col, return_col)

    results_df.to_csv(OUTPUT_DIR / "rf_optuna_results.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "rf_strategy_comparison.csv", index=False)

    best_payload = {
        "best_score": study.best_value,
        "best_params": best_params,
        "objective_priority": "Sharpe Ratio > Cumulative Return > Accuracy",
        "n_trials": N_TRIALS,
        "timeseries_split": {"n_splits": N_SPLITS, "shuffle": False},
    }
    (OUTPUT_DIR / "rf_best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    save_optuna_artifacts(study)
    fi_df = save_feature_importance(best_model, feature_cols)
    plot_cumulative_returns(results_df, date_col)
    plot_cumulative_accuracy(results_df, date_col)

    y_test = test_df[target_col].astype(int)
    y_pred = results_df["predicted"].to_numpy()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("[Random Forest + Optuna] Time-series validation complete.")
    print(f"Rows: total={len(df)}, train={len(train_df)}, test={len(test_df)}")
    print(f"Date column: {date_col} | Target column: {target_col} | Return column: {return_col}")
    print("\n=== Best Hyperparameters ===")
    print(best_params)
    print(f"Best score: {study.best_value:.4f}")
    print("\n=== Classification Metrics (Test) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\n=== Confusion Matrix (Test) ===")
    print(cm)
    print("\n=== Strategy Comparison (Test) ===")
    print(comparison_df.to_string(index=False))
    print("\n=== Top 10 Feature Importance ===")
    print(fi_df.head(10).to_string(index=False))


if __name__ == "__main__":
    # Leakage safety checks:
    # 1) All splits are chronological (shuffle=False, TimeSeriesSplit).
    # 2) Preprocessing is inside pipeline and fitted only on training fold.
    # 3) Validation/test uses strictly future segments.
    main()
