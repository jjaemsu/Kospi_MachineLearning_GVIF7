from __future__ import annotations

import argparse
import base64
import html
import io
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_OUTPUT_DIR = Path("outputs") / "correlation_vif"
DEFAULT_VIF_THRESHOLD = 10.0
DEFAULT_PAIR_THRESHOLD = 0.70


@dataclass
class AnalysisResult:
    label: str
    data: pd.DataFrame
    target_correlation: pd.DataFrame
    pairwise_correlation: pd.DataFrame
    vif: pd.DataFrame
    heatmap_base64: str
    correlation_bar_base64: str
    vif_bar_base64: str


@dataclass
class PruneResult:
    pruned_data: pd.DataFrame
    removed_variables: pd.DataFrame


@dataclass
class ReportRun:
    name: str
    input_path: Path | None
    output_dir: Path
    report_path: Path
    before: AnalysisResult
    prune_result: PruneResult
    after: AnalysisResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build before/after correlation and VIF reports for one or more CSV files. "
            "The script keeps the target column, removes predictors with VIF >= threshold, "
            "and writes HTML plus CSV tables. With multiple inputs it also writes a combined "
            "CV-only report after removing Open/High/Low price columns."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        type=Path,
        help="One or more input CSV paths. Put your later datasets here.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target column for target-correlation analysis. Example: y_KOSPI_Close or Close.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Directory for HTML and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--vif-threshold",
        default=DEFAULT_VIF_THRESHOLD,
        type=float,
        help="Remove predictors whose VIF is at or above this value. Default: 10.",
    )
    parser.add_argument(
        "--pair-threshold",
        default=DEFAULT_PAIR_THRESHOLD,
        type=float,
        help="Highlight variable pairs whose absolute correlation is at or above this value. Default: 0.70.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding. Default: utf-8-sig.",
    )
    parser.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop rows with missing numeric values. By default numeric missing values are median-filled.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._") or "dataset"


def is_ohl_column(column: str, target: str) -> bool:
    if column == target:
        return False

    normalized = column.strip().lower()
    return bool(re.match(r"^(open|high|low|o|h|l)(_|$)", normalized))


def drop_ohl_columns(data: pd.DataFrame, target: str) -> pd.DataFrame:
    drop_columns = [column for column in data.columns if is_ohl_column(column, target)]
    return data.drop(columns=drop_columns)


def clean_numeric_data(raw: pd.DataFrame, target: str, drop_na: bool, source_name: str) -> pd.DataFrame:
    if target not in raw.columns:
        raise ValueError(f"Target column '{target}' was not found in {source_name}. Available columns: {list(raw.columns)}")

    numeric = raw.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' must be numeric for correlation analysis in {source_name}.")

    numeric = numeric.dropna(axis=1, how="all")
    constant_columns = [col for col in numeric.columns if col != target and numeric[col].nunique(dropna=True) <= 1]
    numeric = numeric.drop(columns=constant_columns)
    if target not in numeric.columns:
        raise ValueError(f"Target column '{target}' is constant after cleaning and cannot be analyzed in {source_name}.")

    if drop_na:
        numeric = numeric.dropna(axis=0).copy()
    else:
        numeric = numeric.fillna(numeric.median(numeric_only=True))

    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError(f"No rows remain after numeric cleaning in {source_name}.")
    if numeric.shape[1] < 2:
        raise ValueError(f"At least one numeric predictor plus the target is required in {source_name}.")

    return numeric


def read_numeric_data(input_path: Path, target: str, encoding: str, drop_na: bool) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    raw = pd.read_csv(input_path, encoding=encoding)
    return clean_numeric_data(raw, target, drop_na, str(input_path))


def read_combined_cv_only_data(
    input_paths: list[Path],
    target: str,
    encoding: str,
    drop_na: bool,
) -> pd.DataFrame:
    frames = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
        raw = pd.read_csv(input_path, encoding=encoding)
        numeric = clean_numeric_data(raw, target, drop_na=False, source_name=str(input_path))
        frames.append(drop_ohl_columns(numeric, target))

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return clean_numeric_data(combined, target, drop_na, "combined CV-only data")


def compute_vif(predictors: pd.DataFrame) -> pd.DataFrame:
    if predictors.empty:
        return pd.DataFrame(columns=["Variable", "VIF", "Status"])

    values = predictors.astype(float).to_numpy()
    columns = predictors.columns.tolist()
    rows: list[dict[str, object]] = []

    for idx, column in enumerate(columns):
        y = values[:, idx]
        x = np.delete(values, idx, axis=1)

        if x.shape[1] == 0:
            vif_value = 1.0
        else:
            design = np.column_stack([np.ones(len(predictors)), x])
            coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ coef
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot == 0:
                vif_value = float("inf")
            else:
                r_squared = max(0.0, min(0.999999, 1.0 - ss_res / ss_tot))
                vif_value = 1.0 / (1.0 - r_squared)

        if np.isinf(vif_value) or vif_value >= 10:
            status = "High"
        elif vif_value >= 5:
            status = "Watch"
        else:
            status = "Good"

        rows.append({"Variable": column, "VIF": float(vif_value), "Status": status})

    vif = pd.DataFrame(rows)
    return vif.sort_values(["VIF", "Variable"], ascending=[False, True]).reset_index(drop=True)


def build_target_correlation(data: pd.DataFrame, target: str) -> pd.DataFrame:
    corr = data.corr(numeric_only=True)[target].drop(labels=[target])
    result = pd.DataFrame({"Variable": corr.index, "Correlation": corr.values})
    result["AbsCorrelation"] = result["Correlation"].abs()
    result["Direction"] = np.where(result["Correlation"] >= 0, "Positive", "Negative")
    return result.sort_values(["AbsCorrelation", "Variable"], ascending=[False, True]).reset_index(drop=True)


def build_pairwise_correlation(data: pd.DataFrame, target: str, pair_threshold: float) -> pd.DataFrame:
    predictors = data.drop(columns=[target])
    corr = predictors.corr(numeric_only=True)
    columns = corr.columns.tolist()
    rows: list[dict[str, object]] = []

    for first_idx in range(len(columns)):
        for second_idx in range(first_idx + 1, len(columns)):
            value = float(corr.iloc[first_idx, second_idx])
            abs_value = abs(value)
            rows.append(
                {
                    "Variable1": columns[first_idx],
                    "Variable2": columns[second_idx],
                    "Correlation": value,
                    "AbsCorrelation": abs_value,
                    "Flag": "High" if abs_value >= pair_threshold else "Normal",
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["AbsCorrelation", "Variable1", "Variable2"], ascending=[False, True, True]
    ).reset_index(drop=True)


def fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_heatmap(data: pd.DataFrame, title: str) -> str:
    corr = data.corr(numeric_only=True)
    size = max(7.5, min(18.0, 0.42 * len(corr.columns) + 4.5))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig_to_base64(fig)


def make_correlation_bar(target_correlation: pd.DataFrame, title: str) -> str:
    plot_df = target_correlation.head(20).sort_values("AbsCorrelation", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.28 * len(plot_df) + 1.8)))
    colors = ["#0f766e" if value >= 0 else "#2563eb" for value in plot_df["Correlation"]]
    ax.barh(plot_df["Variable"], plot_df["Correlation"], color=colors)
    ax.axvline(0, color="#64748b", linewidth=1)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Correlation with target")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig_to_base64(fig)


def make_vif_bar(vif: pd.DataFrame, title: str) -> str:
    plot_df = vif.head(20).sort_values("VIF", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(5.2, 0.28 * len(plot_df) + 1.8)))
    colors = []
    for value in plot_df["VIF"]:
        if np.isinf(value) or value >= 10:
            colors.append("#b91c1c")
        elif value >= 5:
            colors.append("#d97706")
        else:
            colors.append("#0f766e")
    ax.barh(plot_df["Variable"], plot_df["VIF"].replace(np.inf, np.nan), color=colors)
    ax.axvline(5, color="#d97706", linewidth=1.2, linestyle="--")
    ax.axvline(10, color="#b91c1c", linewidth=1.2, linestyle="--")
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("VIF")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig_to_base64(fig)


def analyze(data: pd.DataFrame, target: str, label: str, pair_threshold: float) -> AnalysisResult:
    predictors = data.drop(columns=[target])
    target_correlation = build_target_correlation(data, target)
    pairwise_correlation = build_pairwise_correlation(data, target, pair_threshold)
    vif = compute_vif(predictors)
    return AnalysisResult(
        label=label,
        data=data,
        target_correlation=target_correlation,
        pairwise_correlation=pairwise_correlation,
        vif=vif,
        heatmap_base64=make_heatmap(data, f"{label} Correlation Heatmap"),
        correlation_bar_base64=make_correlation_bar(target_correlation, f"{label} Target Correlation"),
        vif_bar_base64=make_vif_bar(vif, f"{label} VIF"),
    )


def prune_by_vif(data: pd.DataFrame, target: str, threshold: float) -> PruneResult:
    pruned = data.copy()
    removed_rows: list[dict[str, object]] = []
    step = 1

    while True:
        predictors = pruned.drop(columns=[target])
        if predictors.shape[1] <= 1:
            break

        vif = compute_vif(predictors)
        highest = vif.iloc[0]
        highest_value = float(highest["VIF"])
        if not (np.isinf(highest_value) or highest_value >= threshold):
            break

        variable = str(highest["Variable"])
        removed_rows.append(
            {
                "Step": step,
                "RemovedVariable": variable,
                "VIFAtRemoval": highest_value,
                "Reason": f"Highest remaining VIF >= {threshold:g}",
            }
        )
        pruned = pruned.drop(columns=[variable])
        step += 1

    removed = pd.DataFrame(
        removed_rows,
        columns=["Step", "RemovedVariable", "VIFAtRemoval", "Reason"],
    )
    return PruneResult(pruned_data=pruned, removed_variables=removed)


def table_html(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    display = frame.copy()
    if max_rows is not None:
        display = display.head(max_rows)

    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].replace([np.inf, -np.inf], np.nan).round(4)

    return display.to_html(index=False, classes="styled-table", border=0, escape=True)


def save_tables(output_dir: Path, prefix: str, result: AnalysisResult) -> None:
    result.target_correlation.to_csv(output_dir / f"{prefix}_target_correlation.csv", index=False, encoding="utf-8-sig")
    result.pairwise_correlation.to_csv(output_dir / f"{prefix}_all_pairwise_correlations.csv", index=False, encoding="utf-8-sig")
    result.vif.to_csv(output_dir / f"{prefix}_vif.csv", index=False, encoding="utf-8-sig")


def render_section(result: AnalysisResult, pair_threshold: float) -> str:
    high_pair_count = int((result.pairwise_correlation["AbsCorrelation"] >= pair_threshold).sum())
    highest_corr = result.target_correlation.iloc[0] if not result.target_correlation.empty else None
    highest_vif = result.vif.iloc[0] if not result.vif.empty else None
    highest_corr_text = (
        f"{highest_corr['Variable']} ({highest_corr['Correlation']:.4f})" if highest_corr is not None else "-"
    )
    highest_vif_text = f"{highest_vif['Variable']} ({highest_vif['VIF']:.4f})" if highest_vif is not None else "-"

    return f"""
    <section class="section">
      <h2>{html.escape(result.label)}</h2>
      <div class="cards">
        <div class="card"><span>Rows</span><strong>{len(result.data):,}</strong></div>
        <div class="card"><span>Predictors</span><strong>{result.data.shape[1] - 1:,}</strong></div>
        <div class="card"><span>Top Target Corr</span><strong>{html.escape(highest_corr_text)}</strong></div>
        <div class="card"><span>Top VIF</span><strong>{html.escape(highest_vif_text)}</strong></div>
        <div class="card"><span>High Pairs</span><strong>{high_pair_count:,}</strong></div>
      </div>
      <div class="visual-grid">
        <img src="data:image/png;base64,{result.heatmap_base64}" alt="{html.escape(result.label)} heatmap">
        <img src="data:image/png;base64,{result.correlation_bar_base64}" alt="{html.escape(result.label)} target correlation chart">
        <img src="data:image/png;base64,{result.vif_bar_base64}" alt="{html.escape(result.label)} VIF chart">
      </div>
      <h3>Target Correlation</h3>
      <div class="table-wrap">{table_html(result.target_correlation)}</div>
      <h3>All Predictor Pairs</h3>
      <p class="note">Every unique predictor pair is included. Rows marked High have |correlation| >= {pair_threshold:g}.</p>
      <div class="table-wrap tall">{table_html(result.pairwise_correlation)}</div>
      <h3>VIF</h3>
      <div class="table-wrap">{table_html(result.vif)}</div>
    </section>
    """


def render_html(
    input_path: Path,
    target: str,
    before: AnalysisResult,
    prune_result: PruneResult,
    after: AnalysisResult,
    pair_threshold: float,
    vif_threshold: float,
) -> str:
    removed_html = (
        table_html(prune_result.removed_variables)
        if not prune_result.removed_variables.empty
        else "<p class='note'>No variables were removed because every VIF was below the threshold.</p>"
    )
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Correlation and VIF Report</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --line: #d9e2ec;
      --accent: #0f766e;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px 18px 48px;
      background: var(--bg);
      color: var(--ink);
      font-family: "Malgun Gothic", "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .wrap {{ width: min(1280px, 100%); margin: 0 auto; }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 18px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; line-height: 1.25; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 24px; }}
    h3 {{ font-size: 18px; margin-top: 22px; }}
    .meta, .note {{ color: var(--muted); font-size: 14px; margin: 0 0 12px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin: 12px 0 18px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: #fbfdff;
      min-width: 0;
    }}
    .card span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
    .card strong {{ display: block; font-size: 17px; word-break: break-word; }}
    .visual-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .visual-grid img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      display: block;
    }}
    .table-wrap {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      max-height: 480px;
      background: white;
    }}
    .table-wrap.tall {{ max-height: 680px; }}
    .styled-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .styled-table th, .styled-table td {{
      border-bottom: 1px solid #edf2f7;
      padding: 9px 11px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    .styled-table thead th {{
      position: sticky;
      top: 0;
      background: #eaf4f2;
      color: #0f172a;
      z-index: 1;
    }}
    @media (max-width: 760px) {{
      body {{ padding: 14px 10px 32px; }}
      .hero, .section {{ padding: 16px; }}
      h1 {{ font-size: 24px; }}
      .visual-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>Correlation and Multicollinearity Report</h1>
      <p class="meta">Source: {html.escape(str(input_path))} | Target: {html.escape(target)} | VIF threshold: {vif_threshold:g} | High pair threshold: |r| >= {pair_threshold:g} | Generated: {generated_at}</p>
      <p class="note">The report computes all unique predictor pairs, target correlations, VIF values, heatmaps, and then repeats the same analysis after iterative removal of predictors whose VIF is at or above the threshold.</p>
    </section>

    <section class="section">
      <h2>Removed Variables</h2>
      {removed_html}
    </section>

    {render_section(before, pair_threshold)}
    {render_section(after, pair_threshold)}
  </main>
</body>
</html>
"""


def write_outputs(
    output_dir: Path,
    input_path: Path | None,
    target: str,
    before: AnalysisResult,
    prune_result: PruneResult,
    after: AnalysisResult,
    pair_threshold: float,
    vif_threshold: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tables(output_dir, "before", before)
    save_tables(output_dir, "after", after)
    prune_result.removed_variables.to_csv(output_dir / "removed_vif_variables.csv", index=False, encoding="utf-8-sig")
    prune_result.pruned_data.to_csv(output_dir / "vif_pruned_dataset.csv", index=False, encoding="utf-8-sig")

    report_path = output_dir / "correlation_vif_report.html"
    report_path.write_text(
        render_html(input_path or Path("combined_cv_only"), target, before, prune_result, after, pair_threshold, vif_threshold),
        encoding="utf-8",
    )
    return report_path


def run_report(
    name: str,
    data: pd.DataFrame,
    output_dir: Path,
    input_path: Path | None,
    target: str,
    pair_threshold: float,
    vif_threshold: float,
) -> ReportRun:
    before = analyze(data, target, "Before VIF Pruning", pair_threshold)
    prune_result = prune_by_vif(data, target, vif_threshold)
    after = analyze(prune_result.pruned_data, target, "After VIF Pruning", pair_threshold)
    report_path = write_outputs(
        output_dir,
        input_path,
        target,
        before,
        prune_result,
        after,
        pair_threshold,
        vif_threshold,
    )
    return ReportRun(
        name=name,
        input_path=input_path,
        output_dir=output_dir,
        report_path=report_path,
        before=before,
        prune_result=prune_result,
        after=after,
    )


def render_index(runs: list[ReportRun], target: str, pair_threshold: float, vif_threshold: float) -> str:
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for run in runs:
        top_corr = run.after.target_correlation.iloc[0] if not run.after.target_correlation.empty else None
        top_vif = run.after.vif.iloc[0] if not run.after.vif.empty else None
        top_corr_text = f"{top_corr['Variable']} ({top_corr['Correlation']:.4f})" if top_corr is not None else "-"
        top_vif_text = f"{top_vif['Variable']} ({top_vif['VIF']:.4f})" if top_vif is not None else "-"
        report_link = html.escape(str(run.report_path.relative_to(run.output_dir.parent)).replace("\\", "/"))
        source = str(run.input_path) if run.input_path is not None else "combined CV-only input"
        rows.append(
            f"""
            <tr>
              <td>{html.escape(run.name)}</td>
              <td>{html.escape(source)}</td>
              <td>{len(run.before.data):,}</td>
              <td>{run.after.data.shape[1] - 1:,}</td>
              <td>{len(run.prune_result.removed_variables):,}</td>
              <td>{html.escape(top_corr_text)}</td>
              <td>{html.escape(top_vif_text)}</td>
              <td><a href="{report_link}">Open report</a></td>
            </tr>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Correlation and VIF Report Index</title>
  <style>
    body {{ margin: 0; padding: 28px 18px 48px; background: #f5f7fb; color: #111827; font-family: "Malgun Gothic", "Segoe UI", sans-serif; line-height: 1.55; }}
    main {{ width: min(1280px, 100%); margin: 0 auto; }}
    section {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 22px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06); }}
    h1 {{ margin: 0 0 10px; font-size: 30px; }}
    p {{ margin: 0 0 16px; color: #64748b; font-size: 14px; }}
    .table-wrap {{ border: 1px solid #d9e2ec; border-radius: 8px; overflow: auto; background: white; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px 12px; text-align: left; vertical-align: top; white-space: nowrap; }}
    thead th {{ position: sticky; top: 0; background: #eaf4f2; color: #0f172a; z-index: 1; }}
    a {{ color: #0f766e; font-weight: 700; text-decoration: none; }}
    @media (max-width: 760px) {{ body {{ padding: 14px 10px 32px; }} section {{ padding: 16px; }} h1 {{ font-size: 24px; }} }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Correlation and VIF Report Index</h1>
      <p>Target: {html.escape(target)} | VIF threshold: {vif_threshold:g} | High pair threshold: |r| >= {pair_threshold:g} | Generated: {generated_at}</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Source</th>
              <th>Rows</th>
              <th>After Predictors</th>
              <th>Removed</th>
              <th>Top After Target Corr</th>
              <th>Top After VIF</th>
              <th>Report</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    runs: list[ReportRun] = []

    for input_path in args.input:
        data = read_numeric_data(input_path, args.target, args.encoding, args.drop_na)
        dataset_dir = args.output_dir / slugify(input_path.stem)
        runs.append(
            run_report(
                input_path.stem,
                data,
                dataset_dir,
                input_path,
                args.target,
                args.pair_threshold,
                args.vif_threshold,
            )
        )

    if len(args.input) > 1:
        combined = read_combined_cv_only_data(args.input, args.target, args.encoding, args.drop_na)
        runs.append(
            run_report(
                "combined_cv_only",
                combined,
                args.output_dir / "combined_cv_only",
                None,
                args.target,
                args.pair_threshold,
                args.vif_threshold,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.html"
    index_path.write_text(render_index(runs, args.target, args.pair_threshold, args.vif_threshold), encoding="utf-8")

    print(f"Created index: {index_path}")
    for run in runs:
        print(f"Created report: {run.report_path}")
        print(f"{run.name} removed variables: {len(run.prune_result.removed_variables)}")


if __name__ == "__main__":
    main()
