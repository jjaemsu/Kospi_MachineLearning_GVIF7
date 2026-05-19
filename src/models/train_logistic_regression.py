"""
KOSPI direction prediction with Logistic Regression.

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
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


TARGET_COLUMN = "y"
DATE_COLUMN = "Date"
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


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    excluded_columns = {TARGET_COLUMN, DATE_COLUMN}
    candidate_columns = [col for col in df.columns if col not in excluded_columns]
    numeric_feature_columns = (
        df[candidate_columns].select_dtypes(include="number").columns.tolist()
    )

    if not numeric_feature_columns:
        raise ValueError("No numeric explanatory variables were found.")

    model_df = df[numeric_feature_columns + [TARGET_COLUMN]].dropna()
    X = model_df[numeric_feature_columns]
    y = model_df[TARGET_COLUMN].astype(int)

    invalid_targets = sorted(set(y.unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must contain only 0 and 1. "
            f"Invalid values: {invalid_targets}"
        )

    return X, y


def split_time_ordered(
    X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(X) * train_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Dataset is too small for an 80/20 time-ordered split.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


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
    original_df: pd.DataFrame,
    y: pd.Series,
    y_pred: pd.Series,
    rise_probability: pd.Series,
    output_path: Path,
) -> None:
    prediction_df = pd.DataFrame(
        {
            "y_true": y.values,
            "y_pred": y_pred.values,
            "rise_probability": rise_probability.values,
        },
        index=y.index,
    )

    if DATE_COLUMN in original_df.columns:
        prediction_df.insert(0, DATE_COLUMN, original_df.loc[y.index, DATE_COLUMN].values)

    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")


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
    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = split_time_ordered(X, y)

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

    save_predictions(df, y_test, y_pred, rise_probability, predictions_path)
    save_coefficients(pipeline, X.columns.tolist(), coefficients_path)
    metrics_path.write_text(metrics_text, encoding="utf-8")

    print("\nSaved files")
    print(f"- Predictions: {predictions_path}")
    print(f"- Coefficients: {coefficients_path}")
    print(f"- Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
