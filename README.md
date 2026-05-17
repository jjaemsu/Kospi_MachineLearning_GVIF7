# Kospi_MachineLearning_GVIF7

This project contains machine learning models and trading strategies.

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
