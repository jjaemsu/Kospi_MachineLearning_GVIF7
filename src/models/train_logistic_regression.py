"""
KOSPI next-day direction prediction with Logistic Regression.

Run after placing your preprocessed and VIF-pruned CSV file in the project.

Example:
    python src/models/train_logistic_regression.py --input data/vif_pruned_dataset.csv

Optional:
    python src/models/train_logistic_regression.py \
        --input data/your_file.csv \
        --output-dir outputs/logistic_regression

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
RETURN_COLUMN_CANDIDATES = ("Return", "Log_Return")
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
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in rows. Default: 1 means X_t predicts y_t+1.",
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Put your CSV file in the project and pass it with --input."
        )

    df = pd.read_csv(input_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    return df


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


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def format_metrics(y_test: pd.Series, y_pred: pd.Series) -> str:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return (
        "KOSPI Logistic Regression Evaluation\n"
        "====================================\n"
        f"Accuracy : {accuracy:.6f}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall   : {recall:.6f}\n"
        f"F1-score : {f1:.6f}\n\n"
        "Confusion Matrix\n"
        "----------------\n"
        f"{cm}\n\n"
        "Classification Report\n"
        "---------------------\n"
        f"{report}"
    )


def save_predictions(
    metadata: pd.DataFrame,
    y: pd.Series,
    y_pred: pd.Series,
    rise_probability: pd.Series,
    output_path: Path,
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

    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return prediction_df


def save_return_comparison(
    prediction_df: pd.DataFrame,
    output_dir: Path,
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

    if return_type == "Log_Return":
        plot_df["buy_and_hold_cumulative_return"] = np.exp(
            plot_df["target_return"].cumsum()
        ) - 1
        plot_df["strategy_return"] = np.where(
            plot_df["y_pred"] == 1,
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
            plot_df["y_pred"] == 1,
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
        label="Model strategy",
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
    X_train, X_test, y_train, y_test, test_metadata = split_time_ordered(X, y, metadata)

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    rise_probability = pd.Series(
        pipeline.predict_proba(X_test)[:, 1],
        index=y_test.index,
        name="rise_probability",
    )
    y_pred = (rise_probability >= args.threshold).astype(int)

    metrics_text = format_metrics(y_test, y_pred)
    print(metrics_text)

    predictions_path = output_dir / "logistic_regression_test_predictions.csv"
    coefficients_path = output_dir / "logistic_regression_coefficients.csv"
    metrics_path = output_dir / "logistic_regression_metrics.txt"

    prediction_df = save_predictions(
        test_metadata,
        y_test,
        y_pred,
        rise_probability,
        predictions_path,
    )
    save_coefficients(pipeline, X.columns.tolist(), coefficients_path)
    metrics_path.write_text(metrics_text, encoding="utf-8")
    return_chart_path = save_return_comparison(prediction_df, output_dir)
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
