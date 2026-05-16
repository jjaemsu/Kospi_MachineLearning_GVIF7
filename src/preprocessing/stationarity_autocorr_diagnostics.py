from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


DEFAULT_OUTPUT_DIR = Path("outputs") / "stationarity_autocorr"


plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ADF stationarity tests, choose a recommended differencing transform, "
            "and generate ACF/PACF diagnostics."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--target", default="Close", help="Target series column to diagnose")
    parser.add_argument("--date-col", default="Date", help="Date column name")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Output directory")
    parser.add_argument("--alpha", default=0.05, type=float, help="ADF significance level")
    parser.add_argument("--max-lags", default=40, type=int, help="Max lag for ACF/PACF plots")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    return parser.parse_args()


def read_series(input_path: Path, target: str, date_col: str, encoding: str) -> pd.Series:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    df = pd.read_csv(input_path, encoding=encoding)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV")

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    series = pd.to_numeric(df[target], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Target column '{target}' has no numeric values")
    return series


def build_transforms(series: pd.Series) -> dict[str, pd.Series]:
    transforms: dict[str, pd.Series] = {
        "none": series.copy(),
        "diff_1": series.diff().dropna(),
    }

    if (series > 0).all():
        transforms["log_diff_1"] = np.log(series).diff().dropna()

    return transforms


def run_adf(series: pd.Series) -> dict[str, float | str]:
    stat, pvalue, usedlag, nobs, critical_values, _ = adfuller(series, autolag="AIC")
    return {
        "adf_stat": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(usedlag),
        "n_obs": int(nobs),
        "crit_1pct": float(critical_values["1%"]),
        "crit_5pct": float(critical_values["5%"]),
        "crit_10pct": float(critical_values["10%"]),
    }


def recommend_transform(adf_table: pd.DataFrame, alpha: float) -> str:
    stationary = adf_table[adf_table["p_value"] < alpha]
    if stationary.empty:
        return "none"

    priority = ["none", "diff_1", "log_diff_1"]
    for name in priority:
        if name in stationary["transform"].values:
            return name

    return str(stationary.iloc[0]["transform"])


def plot_acf_pacf(series: pd.Series, output_path: Path, max_lags: int) -> None:
    lags = min(max_lags, max(5, len(series) // 3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_acf(series, ax=axes[0], lags=lags, alpha=0.05, title="ACF")
    plot_pacf(series, ax=axes[1], lags=lags, alpha=0.05, method="ywm", title="PACF")
    fig.suptitle("Autocorrelation Diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_acf_pacf_table(series: pd.Series, output_path: Path, max_lags: int) -> None:
    from statsmodels.tsa.stattools import acf, pacf

    lags = min(max_lags, max(5, len(series) // 3))
    acf_values = acf(series, nlags=lags, fft=True)
    pacf_values = pacf(series, nlags=lags, method="ywm")
    rows = [{"lag": lag, "acf": float(acf_values[lag]), "pacf": float(pacf_values[lag])} for lag in range(lags + 1)]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_series = read_series(args.input, args.target, args.date_col, args.encoding)
    transforms = build_transforms(raw_series)

    adf_rows = []
    for name, values in transforms.items():
        if len(values) < 30:
            continue
        metrics = run_adf(values)
        metrics["transform"] = name
        metrics["n_series"] = int(len(values))
        adf_rows.append(metrics)

    if not adf_rows:
        raise ValueError("No transform had enough data points for ADF testing")

    adf_table = pd.DataFrame(adf_rows)[
        [
            "transform",
            "n_series",
            "adf_stat",
            "p_value",
            "used_lag",
            "n_obs",
            "crit_1pct",
            "crit_5pct",
            "crit_10pct",
        ]
    ].sort_values("p_value", ascending=True)

    recommended = recommend_transform(adf_table, args.alpha)
    recommended_series = transforms[recommended]

    adf_table["is_stationary_at_alpha"] = adf_table["p_value"] < args.alpha
    adf_table.to_csv(args.output_dir / "adf_results.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "input_csv": str(args.input),
                "target": args.target,
                "alpha": args.alpha,
                "recommended_transform": recommended,
                "recommended_length": len(recommended_series),
            }
        ]
    )
    summary.to_csv(args.output_dir / "stationarity_summary.csv", index=False, encoding="utf-8-sig")

    transformed_df = pd.DataFrame(
        {
            args.target: raw_series,
            f"{args.target}_diff_1": transforms.get("diff_1"),
            f"{args.target}_log_diff_1": transforms.get("log_diff_1"),
        }
    )
    transformed_df.to_csv(args.output_dir / "transformed_target_series.csv", index=False, encoding="utf-8-sig")

    plot_acf_pacf(recommended_series, args.output_dir / "acf_pacf_recommended.png", args.max_lags)
    save_acf_pacf_table(recommended_series, args.output_dir / "acf_pacf_values_recommended.csv", args.max_lags)

    print(f"Created: {args.output_dir / 'adf_results.csv'}")
    print(f"Created: {args.output_dir / 'stationarity_summary.csv'}")
    print(f"Recommended transform: {recommended}")


if __name__ == "__main__":
    main()
