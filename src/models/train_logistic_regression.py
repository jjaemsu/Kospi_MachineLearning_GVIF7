"""
KOSPI next-day direction prediction with Logistic Regression.

Run after placing your preprocessed and VIF-pruned CSV file in the project.

Example:
    python src/models/train_logistic_regression.py --input data/vif_pruned_dataset.csv

Optional:
    python src/models/train_logistic_regression.py \
        --input data/your_file.csv \
        --output-dir outputs/logistic_regression
    python src/models/train_logistic_regression.py \
        --input data/your_file.csv \
        --validation-mode walk_forward \
        --initial-train-ratio 0.8

Input requirements:
    - Target column: y
    - y values: rise = 1, fall = 0
    - Date column is allowed and excluded from X
    - Only numeric explanatory variables are used as X
    - X on day t predicts y on day t+1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_COLUMN = "y"
DATE_COLUMN = "Date"
PREDICTION_TARGET_COLUMN = "y_next"
RETURN_COLUMN_CANDIDATES = ("Return", "Log_Return", "price_Log_Return")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression model for KOSPI up/down prediction."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the preprocessed and VIF-pruned CSV file to train on.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/logistic_regression",
        help="Directory where predictions, coefficients, and metrics are saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying rise(1). Default: 0.5",
    )
    parser.add_argument(
        "--trade-entry-threshold",
        type=float,
        default=0.52,
        help=(
            "Rise probability threshold for entering the long-only strategy. "
            "Default: 0.52 means probabilities below 0.52 stay in cash."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in rows. Default: 1 means X_t predicts y_t+1.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["walk_forward", "holdout"],
        default="walk_forward",
        help=(
            "Validation method. walk_forward retrains on all past rows before each "
            "test prediction. holdout trains once on the first train ratio. "
            "Default: walk_forward."
        ),
    )
    parser.add_argument(
        "--initial-train-ratio",
        type=float,
        default=0.8,
        help=(
            "Initial train ratio for walk-forward or holdout split. "
            "Default: 0.8."
        ),
    )
    parser.add_argument(
        "--initial-train-size",
        type=int,
        default=None,
        help=(
            "Optional fixed number of initial training rows for walk-forward. "
            "If set, this overrides --initial-train-ratio."
        ),
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting for LogisticRegression. Default: balanced.",
    )
    parser.add_argument(
        "--c-values",
        default="1.0",
        help="Comma-separated inverse regularization strengths. Default: 1.0.",
    )
    parser.add_argument(
        "--penalties",
        default="l2",
        help="Comma-separated penalties among l1,l2. Default: l2.",
    )
    parser.add_argument(
        "--threshold-values",
        default=None,
        help=(
            "Optional comma-separated prediction thresholds to tune. "
            "If omitted, --threshold is used."
        ),
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help=(
            "Tune class_weight, C, penalty, and threshold on a time-ordered "
            "validation window before the final test window."
        ),
    )
    parser.add_argument(
        "--tuning-validation-ratio",
        type=float,
        default=0.2,
        help="Validation ratio inside the pre-test window for tuning. Default: 0.2.",
    )
    parser.add_argument(
        "--min-validation-balanced-accuracy",
        type=float,
        default=0.52,
        help=(
            "Minimum validation balanced accuracy required to accept tuned "
            "hyperparameters. Default: 0.52."
        ),
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Put your CSV file in the project and pass it with --input."
        )

    df = pd.read_csv(input_path)
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    return df


def parse_float_list(raw_value: str) -> list[float]:
    values = [float(value.strip()) for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def parse_str_list(raw_value: str, allowed_values: set[str]) -> list[str]:
    values = [value.strip().lower() for value in raw_value.split(",") if value.strip()]
    invalid_values = sorted(set(values) - allowed_values)
    if invalid_values:
        raise ValueError(f"Invalid values {invalid_values}. Allowed: {sorted(allowed_values)}")
    if not values:
        raise ValueError("At least one value is required.")
    return values


def build_model_dataset(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if horizon < 1:
        raise ValueError("--horizon must be at least 1.")

    excluded_columns = {TARGET_COLUMN, DATE_COLUMN}
    candidate_columns = [col for col in df.columns if col not in excluded_columns]
    numeric_feature_columns = (
        df[candidate_columns].select_dtypes(include="number").columns.tolist()
    )

    if not numeric_feature_columns:
        raise ValueError("No numeric explanatory variables were found.")

    invalid_targets = sorted(set(df[TARGET_COLUMN].dropna().astype(int).unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain only 0 and 1. "
            f"Invalid values: {invalid_targets}"
        )

    model_df = df[numeric_feature_columns + [TARGET_COLUMN]].copy()
    model_df[PREDICTION_TARGET_COLUMN] = model_df[TARGET_COLUMN].shift(-horizon)

    metadata = pd.DataFrame(index=df.index)
    if DATE_COLUMN in df.columns:
        metadata["input_date"] = df[DATE_COLUMN]
        metadata["target_date"] = df[DATE_COLUMN].shift(-horizon)

    return_column = next(
        (col for col in RETURN_COLUMN_CANDIDATES if col in df.columns),
        None,
    )
    if return_column:
        metadata["target_return"] = df[return_column].shift(-horizon)
        metadata["return_type"] = return_column

    model_df = model_df.join(metadata)
    required_columns = numeric_feature_columns + [PREDICTION_TARGET_COLUMN]
    model_df = model_df.dropna(subset=required_columns)

    X = model_df[numeric_feature_columns]
    y = model_df[PREDICTION_TARGET_COLUMN].astype(int)
    metadata_columns = [
        col
        for col in ["input_date", "target_date", "target_return", "return_type"]
        if col in model_df.columns
    ]
    prediction_metadata = model_df[metadata_columns]

    return X, y, prediction_metadata


def split_time_ordered(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    split_index = int(len(X) * train_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Dataset is too small for an 80/20 time-ordered split.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    test_metadata = metadata.iloc[split_index:]

    return X_train, X_test, y_train, y_test, test_metadata


def get_initial_train_size(
    n_rows: int,
    initial_train_ratio: float,
    initial_train_size: int | None,
) -> int:
    if initial_train_size is not None:
        split_index = initial_train_size
    else:
        if not 0 < initial_train_ratio < 1:
            raise ValueError("--initial-train-ratio must be between 0 and 1.")
        split_index = int(n_rows * initial_train_ratio)

    if split_index <= 0 or split_index >= n_rows:
        raise ValueError(
            "Initial training size must leave at least one row for both "
            "training and walk-forward testing."
        )

    return split_index


def normalize_class_weight(class_weight: str) -> str | None:
    return None if class_weight == "none" else class_weight


def make_pipeline(
    class_weight: str = "balanced",
    c_value: float = 1.0,
    penalty: str = "l2",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight=normalize_class_weight(class_weight),
                    max_iter=1000,
                    penalty=penalty,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def run_holdout_validation(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    threshold: float,
    train_ratio: float,
    class_weight: str,
    c_value: float,
    penalty: str,
) -> tuple[pd.DataFrame, Pipeline]:
    X_train, X_test, y_train, y_test, test_metadata = split_time_ordered(
        X, y, metadata, train_ratio=train_ratio
    )

    pipeline = make_pipeline(class_weight, c_value, penalty)
    pipeline.fit(X_train, y_train)

    rise_probability = pd.Series(
        pipeline.predict_proba(X_test)[:, 1],
        index=y_test.index,
        name="rise_probability",
    )
    y_pred = (rise_probability >= threshold).astype(int)

    prediction_df = build_prediction_frame(
        test_metadata,
        y_test,
        y_pred,
        rise_probability,
    )

    return prediction_df, pipeline


def run_walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    threshold: float,
    initial_train_ratio: float,
    initial_train_size: int | None,
    class_weight: str,
    c_value: float,
    penalty: str,
) -> tuple[pd.DataFrame, Pipeline]:
    split_index = get_initial_train_size(
        len(X),
        initial_train_ratio,
        initial_train_size,
    )

    prediction_rows = []
    final_pipeline: Pipeline | None = None

    for test_position in range(split_index, len(X)):
        X_train = X.iloc[:test_position]
        y_train = y.iloc[:test_position]
        X_test = X.iloc[[test_position]]
        y_test = y.iloc[[test_position]]
        test_metadata = metadata.iloc[[test_position]]

        pipeline = make_pipeline(class_weight, c_value, penalty)
        pipeline.fit(X_train, y_train)
        final_pipeline = pipeline

        rise_probability = pd.Series(
            pipeline.predict_proba(X_test)[:, 1],
            index=y_test.index,
            name="rise_probability",
        )
        y_pred = (rise_probability >= threshold).astype(int)

        prediction_rows.append(
            build_prediction_frame(
                test_metadata,
                y_test,
                y_pred,
                rise_probability,
            )
        )

    if final_pipeline is None:
        raise ValueError("Walk-forward validation did not produce any predictions.")

    prediction_df = pd.concat(prediction_rows, ignore_index=True)
    return prediction_df, final_pipeline


def evaluate_binary_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_probability_rmse(
    y_true: pd.Series,
    rise_probability: pd.Series,
) -> float:
    return float(np.sqrt(np.mean((y_true.to_numpy() - rise_probability.to_numpy()) ** 2)))


def build_baseline_metrics(y_true: pd.Series) -> pd.DataFrame:
    y_true = y_true.reset_index(drop=True).astype(int)
    majority_class = int(y_true.mode().iloc[0])
    previous_direction = y_true.shift(1).fillna(majority_class).astype(int)
    baselines = {
        "always_up": pd.Series(1, index=y_true.index),
        "always_down": pd.Series(0, index=y_true.index),
        "majority_class": pd.Series(majority_class, index=y_true.index),
        "previous_observed_direction": previous_direction,
    }

    rows = []
    for name, y_pred in baselines.items():
        row = {"baseline": name}
        row.update(evaluate_binary_predictions(y_true, y_pred))
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_return_summary(
    prediction_df: pd.DataFrame,
    trade_entry_threshold: float,
) -> dict[str, float] | None:
    if "target_return" not in prediction_df.columns:
        return None

    return_type = (
        prediction_df["return_type"].iloc[0]
        if "return_type" in prediction_df.columns
        else "Return"
    )
    plot_df = prediction_df.copy()
    plot_df["target_return"] = pd.to_numeric(
        plot_df["target_return"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["target_return"])
    if plot_df.empty:
        return None

    plot_df["strategy_position"] = np.where(
        plot_df["rise_probability"] >= trade_entry_threshold,
        1,
        0,
    )
    plot_df["strategy_return"] = np.where(
        plot_df["strategy_position"] == 1,
        plot_df["target_return"],
        0.0,
    )

    if return_type in {"Log_Return", "price_Log_Return"}:
        buy_and_hold = np.exp(plot_df["target_return"].sum()) - 1
        strategy = np.exp(plot_df["strategy_return"].sum()) - 1
    else:
        buy_and_hold = (1 + plot_df["target_return"]).prod() - 1
        strategy = (1 + plot_df["strategy_return"]).prod() - 1

    return {
        "buy_and_hold_cumulative_return": buy_and_hold,
        "model_strategy_cumulative_return": strategy,
        "trade_rate": float(plot_df["strategy_position"].mean()),
    }


def format_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
    rise_probability: pd.Series,
) -> str:
    metrics = evaluate_binary_predictions(y_test, y_pred)
    rmse = calculate_probability_rmse(y_test, rise_probability)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    baseline_df = build_baseline_metrics(y_test)

    return (
        "KOSPI Logistic Regression Evaluation\n"
        "====================================\n"
        f"Accuracy          : {metrics['accuracy']:.6f}\n"
        f"Balanced Accuracy : {metrics['balanced_accuracy']:.6f}\n"
        f"Precision         : {metrics['precision']:.6f}\n"
        f"Recall            : {metrics['recall']:.6f}\n"
        f"F1-score          : {metrics['f1']:.6f}\n"
        f"Probability RMSE  : {rmse:.6f}\n\n"
        "Baseline Comparison\n"
        "-------------------\n"
        f"{baseline_df.to_string(index=False)}\n\n"
        "Confusion Matrix\n"
        "----------------\n"
        f"{cm}\n\n"
        "Classification Report\n"
        "---------------------\n"
        f"{report}"
    )


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    test_start: int,
    validation_ratio: float,
    class_weights: list[str],
    c_values: list[float],
    penalties: list[str],
    thresholds: list[float],
    trade_entry_threshold: float,
    min_validation_balanced_accuracy: float,
    output_dir: Path,
) -> dict[str, object]:
    if not 0 < validation_ratio < 1:
        raise ValueError("--tuning-validation-ratio must be between 0 and 1.")

    validation_start = int(test_start * (1 - validation_ratio))
    if validation_start <= 0 or validation_start >= test_start:
        raise ValueError("Tuning split leaves no train or validation rows.")

    X_tune = X.iloc[:test_start]
    y_tune = y.iloc[:test_start]
    metadata_tune = metadata.iloc[:test_start]

    rows = []
    best_row: dict[str, object] | None = None
    for class_weight in class_weights:
        for c_value in c_values:
            for penalty in penalties:
                for threshold in thresholds:
                    prediction_df, _ = run_walk_forward_validation(
                        X_tune,
                        y_tune,
                        metadata_tune,
                        threshold,
                        initial_train_ratio=0.8,
                        initial_train_size=validation_start,
                        class_weight=class_weight,
                        c_value=c_value,
                        penalty=penalty,
                    )
                    metrics = evaluate_binary_predictions(
                        prediction_df["y_true"],
                        prediction_df["y_pred"],
                    )
                    rmse = calculate_probability_rmse(
                        prediction_df["y_true"],
                        prediction_df["rise_probability"],
                    )
                    return_summary = calculate_return_summary(
                        prediction_df,
                        trade_entry_threshold,
                    )
                    row = {
                        "class_weight": class_weight,
                        "C": c_value,
                        "penalty": penalty,
                        "threshold": threshold,
                        "validation_rows": len(prediction_df),
                        "probability_rmse": rmse,
                        **metrics,
                    }
                    if return_summary:
                        row.update(return_summary)
                    rows.append(row)

                    if best_row is None or (
                        metrics["balanced_accuracy"],
                        metrics["f1"],
                    ) > (
                        float(best_row["balanced_accuracy"]),
                        float(best_row["f1"]),
                    ):
                        best_row = row

    tuning_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1", "accuracy"],
        ascending=False,
    )
    tuning_df.to_csv(output_dir / "hyperparameter_tuning_results.csv", index=False)

    if best_row is None:
        raise ValueError("Hyperparameter tuning did not produce any result.")

    best_row["accepted"] = (
        float(best_row["balanced_accuracy"]) >= min_validation_balanced_accuracy
    )
    best_row["min_validation_balanced_accuracy"] = min_validation_balanced_accuracy

    return best_row


def build_prediction_frame(
    metadata: pd.DataFrame,
    y: pd.Series,
    y_pred: pd.Series,
    rise_probability: pd.Series,
) -> pd.DataFrame:
    prediction_df = pd.DataFrame(
        {
            "y_true": y.values,
            "y_pred": y_pred.values,
            "rise_probability": rise_probability.values,
        },
        index=y.index,
    )

    if not metadata.empty:
        prediction_df = pd.concat(
            [metadata.reset_index(drop=True), prediction_df.reset_index(drop=True)],
            axis=1,
        )

    return prediction_df


def save_predictions(prediction_df: pd.DataFrame, output_path: Path) -> None:
    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_return_comparison(
    prediction_df: pd.DataFrame,
    output_dir: Path,
    trade_entry_threshold: float,
) -> Path | None:
    if "target_return" not in prediction_df.columns:
        return None

    return_type = (
        prediction_df["return_type"].iloc[0]
        if "return_type" in prediction_df.columns
        else "Return"
    )
    plot_df = prediction_df.copy()
    plot_df["target_return"] = pd.to_numeric(
        plot_df["target_return"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["target_return"])
    if plot_df.empty:
        return None

    plot_df["strategy_position"] = np.where(
        plot_df["rise_probability"] >= trade_entry_threshold,
        1,
        0,
    )

    if return_type in {"Log_Return", "price_Log_Return"}:
        plot_df["buy_and_hold_cumulative_return"] = np.exp(
            plot_df["target_return"].cumsum()
        ) - 1
        plot_df["strategy_return"] = np.where(
            plot_df["strategy_position"] == 1,
            plot_df["target_return"],
            0.0,
        )
        plot_df["model_strategy_cumulative_return"] = np.exp(
            plot_df["strategy_return"].cumsum()
        ) - 1
    else:
        plot_df["buy_and_hold_cumulative_return"] = (
            1 + plot_df["target_return"]
        ).cumprod() - 1
        plot_df["strategy_return"] = np.where(
            plot_df["strategy_position"] == 1,
            plot_df["target_return"],
            0.0,
        )
        plot_df["model_strategy_cumulative_return"] = (
            1 + plot_df["strategy_return"]
        ).cumprod() - 1

    csv_path = output_dir / "return_comparison.csv"
    plot_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    x_values = (
        pd.to_datetime(plot_df["target_date"])
        if "target_date" in plot_df.columns
        else plot_df.index
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        x_values,
        plot_df["buy_and_hold_cumulative_return"],
        label="Buy and hold",
        linewidth=2,
    )
    ax.plot(
        x_values,
        plot_df["model_strategy_cumulative_return"],
        label=f"Model strategy (P(up) >= {trade_entry_threshold:.2f})",
        linewidth=2,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Cumulative Return Comparison")
    ax.set_xlabel("Target date")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    chart_path = output_dir / "cumulative_return_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def save_prediction_rate(
    prediction_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    plot_df = prediction_df.copy()
    plot_df["is_correct"] = (plot_df["y_true"] == plot_df["y_pred"]).astype(int)
    plot_df["prediction_count"] = np.arange(1, len(plot_df) + 1)
    plot_df["correct_count"] = plot_df["is_correct"].cumsum()
    plot_df["model_accuracy_to_date"] = (
        plot_df["correct_count"] / plot_df["prediction_count"]
    )

    csv_path = output_dir / "prediction_rate.csv"
    plot_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    x_values = (
        pd.to_datetime(plot_df["target_date"])
        if "target_date" in plot_df.columns
        else plot_df.index
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        x_values,
        plot_df["model_accuracy_to_date"],
        label="Model accuracy to date",
        linewidth=2,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Rate")
    ax.set_xlabel("Target date")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    chart_path = output_dir / "prediction_rate.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def save_coefficients(
    pipeline: Pipeline, feature_names: list[str], output_path: Path
) -> None:
    model = pipeline.named_steps["model"]
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)

    intercept_row = pd.DataFrame(
        {"feature": ["intercept"], "coefficient": [model.intercept_[0]]}
    )
    coef_df = pd.concat([coef_df, intercept_row], ignore_index=True)
    coef_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    X, y, metadata = build_model_dataset(df, args.horizon)
    c_values = parse_float_list(args.c_values)
    penalties = parse_str_list(args.penalties, {"l1", "l2"})
    thresholds = (
        parse_float_list(args.threshold_values)
        if args.threshold_values
        else [args.threshold]
    )
    class_weights = (
        ["none", "balanced"]
        if args.tune_hyperparameters
        else [args.class_weight]
    )

    selected_class_weight = args.class_weight
    selected_c_value = 1.0 if args.tune_hyperparameters else c_values[0]
    selected_penalty = "l2" if args.tune_hyperparameters else penalties[0]
    selected_threshold = args.threshold

    if args.tune_hyperparameters:
        test_start = get_initial_train_size(
            len(X),
            args.initial_train_ratio,
            args.initial_train_size,
        )
        best_params = tune_hyperparameters(
            X,
            y,
            metadata,
            test_start,
            args.tuning_validation_ratio,
            class_weights,
            c_values,
            penalties,
            thresholds,
            args.trade_entry_threshold,
            args.min_validation_balanced_accuracy,
            output_dir,
        )
        if bool(best_params["accepted"]):
            selected_class_weight = str(best_params["class_weight"])
            selected_c_value = float(best_params["C"])
            selected_penalty = str(best_params["penalty"])
            selected_threshold = float(best_params["threshold"])
        print("\nSelected hyperparameters from pre-test validation")
        print("-----------------------------------------------")
        print(f"best validation class_weight      : {best_params['class_weight']}")
        print(f"best validation C                 : {best_params['C']}")
        print(f"best validation penalty           : {best_params['penalty']}")
        print(f"best validation threshold         : {best_params['threshold']}")
        print(f"best validation balanced accuracy : {best_params['balanced_accuracy']:.6f}")
        print(f"minimum required balanced accuracy: {best_params['min_validation_balanced_accuracy']:.6f}")
        print(f"accepted tuned hyperparameters    : {best_params['accepted']}")
        if not bool(best_params["accepted"]):
            print("Using CLI/default model settings for final test instead.")

    if args.validation_mode == "walk_forward":
        prediction_df, pipeline = run_walk_forward_validation(
            X,
            y,
            metadata,
            selected_threshold,
            args.initial_train_ratio,
            args.initial_train_size,
            selected_class_weight,
            selected_c_value,
            selected_penalty,
        )
    else:
        prediction_df, pipeline = run_holdout_validation(
            X,
            y,
            metadata,
            selected_threshold,
            args.initial_train_ratio,
            selected_class_weight,
            selected_c_value,
            selected_penalty,
        )

    metrics_text = format_metrics(
        prediction_df["y_true"],
        prediction_df["y_pred"],
        prediction_df["rise_probability"],
    )
    return_summary = calculate_return_summary(
        prediction_df,
        args.trade_entry_threshold,
    )
    if return_summary:
        metrics_text += (
            "\n\nReturn Summary\n"
            "--------------\n"
            f"Buy and hold cumulative return : "
            f"{return_summary['buy_and_hold_cumulative_return']:.6f}\n"
            f"Model strategy cumulative return: "
            f"{return_summary['model_strategy_cumulative_return']:.6f}\n"
            f"Trade rate                      : "
            f"{return_summary['trade_rate']:.6f}"
        )
    metrics_text += (
        "\n\nSelected Model Settings\n"
        "-----------------------\n"
        f"validation_mode       : {args.validation_mode}\n"
        f"class_weight          : {selected_class_weight}\n"
        f"C                     : {selected_c_value}\n"
        f"penalty               : {selected_penalty}\n"
        f"prediction_threshold  : {selected_threshold}\n"
        f"trade_entry_threshold : {args.trade_entry_threshold}"
    )
    print(metrics_text)

    predictions_path = output_dir / "logistic_regression_test_predictions.csv"
    coefficients_path = output_dir / "logistic_regression_coefficients.csv"
    metrics_path = output_dir / "logistic_regression_metrics.txt"

    save_predictions(prediction_df, predictions_path)
    save_coefficients(pipeline, X.columns.tolist(), coefficients_path)
    metrics_path.write_text(metrics_text, encoding="utf-8")
    return_chart_path = save_return_comparison(
        prediction_df,
        output_dir,
        args.trade_entry_threshold,
    )
    prediction_rate_chart_path = save_prediction_rate(
        prediction_df,
        output_dir,
    )

    print("\nSaved files")
    print(f"- Predictions: {predictions_path}")
    print(f"- Coefficients: {coefficients_path}")
    print(f"- Metrics: {metrics_path}")
    if return_chart_path:
        print(f"- Return chart: {return_chart_path}")
        print(f"- Return data: {output_dir / 'return_comparison.csv'}")
    else:
        print("- Return chart: skipped because no Return or Log_Return column was found.")
    print(f"- Prediction rate chart: {prediction_rate_chart_path}")
    print(f"- Prediction rate data: {output_dir / 'prediction_rate.csv'}")


if __name__ == "__main__":
    main()
