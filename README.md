# Kospi_MachineLearning_GVIF7

This project contains machine learning models and trading strategies.

## Random Forest (KOSPI Direction)

Train and evaluate the Random Forest classifier with lag1 stationary inputs:

```powershell
python scripts/train_random_forest.py
```

Outputs are saved under `outputs/`:

- `rf_cumulative_return_comparison.png`
- `rf_cumulative_prediction_accuracy.png`
- `rf_feature_importance.csv`
- `rf_prediction_results.csv`

## Correlation and VIF report

`src/evaluation/correlation_vif_report.py` reads one or more CSV datasets, analyzes target correlations and predictor multicollinearity, removes predictors with high VIF, and writes before/after reports. When multiple CSVs are provided, it also creates a combined CV-only report that removes Open/High/Low price columns before analysis.

Run after placing your dataset in the project:

```powershell
python src/evaluation/correlation_vif_report.py --input data/your_dataset.csv --target y_KOSPI_Close
```

Multiple CSVs:

```powershell
python src/evaluation/correlation_vif_report.py `
  --input data/company_a.csv data/company_b.csv data/company_c.csv `
  --target Close
```

Useful options:

```powershell
python src/evaluation/correlation_vif_report.py `
  --input data/company_a.csv data/company_b.csv `
  --target Close `
  --output-dir outputs/correlation_vif `
  --vif-threshold 10 `
  --pair-threshold 0.70
```

Outputs:

- `index.html`: summary page linking to each generated report.
- `<dataset_name>/correlation_vif_report.html`: before/after HTML report for each input CSV.
- `combined_cv_only/correlation_vif_report.html`: combined report for all inputs after dropping Open/High/Low columns. This is created only when two or more CSVs are provided.
- `before_all_pairwise_correlations.csv` and `after_all_pairwise_correlations.csv`: every unique predictor pair.
- `before_target_correlation.csv` and `after_target_correlation.csv`: target-correlation tables.
- `before_vif.csv` and `after_vif.csv`: VIF tables.
- `removed_vif_variables.csv`: variables removed because VIF was at or above the threshold.
- `vif_pruned_dataset.csv`: cleaned dataset after VIF pruning.

## Stationarity (ADF) and Autocorrelation (ACF/PACF)

Use `src/preprocessing/stationarity_autocorr_diagnostics.py` to run ADF tests on the target series (`none`, `1st diff`, `log 1st diff`), auto-select a recommended transform, and export ACF/PACF diagnostics.

```powershell
python src/preprocessing/stationarity_autocorr_diagnostics.py --input data/merged_data.csv --target Close
```

Outputs under `outputs/stationarity_autocorr`:

- `adf_results.csv`: ADF statistics for each transform.
- `stationarity_summary.csv`: recommended transform at selected alpha.
- `transformed_target_series.csv`: original + differenced series columns.
- `acf_pacf_recommended.png`: ACF/PACF chart for recommended transform.
- `acf_pacf_values_recommended.csv`: lag-wise ACF/PACF values.
