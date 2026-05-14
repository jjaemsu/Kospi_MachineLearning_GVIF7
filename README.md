# Kospi_MachineLearning_GVIF7

This project contains machine learning models and trading strategies.

## Correlation and VIF report

`src/evaluation/correlation_vif_report.py` reads a CSV dataset, analyzes target correlations and predictor multicollinearity, removes predictors with high VIF, and writes before/after reports.

Run after placing your dataset in the project:

```powershell
python src/evaluation/correlation_vif_report.py --input data/your_dataset.csv --target y_KOSPI_Close
```

Useful options:

```powershell
python src/evaluation/correlation_vif_report.py `
  --input data/your_dataset.csv `
  --target y_KOSPI_Close `
  --output-dir outputs/correlation_vif `
  --vif-threshold 10 `
  --pair-threshold 0.70
```

Outputs:

- `correlation_vif_report.html`: before/after HTML report with heatmaps, charts, full pair tables, target correlations, and VIF tables.
- `before_all_pairwise_correlations.csv` and `after_all_pairwise_correlations.csv`: every unique predictor pair.
- `before_target_correlation.csv` and `after_target_correlation.csv`: target-correlation tables.
- `before_vif.csv` and `after_vif.csv`: VIF tables.
- `removed_vif_variables.csv`: variables removed because VIF was at or above the threshold.
- `vif_pruned_dataset.csv`: cleaned dataset after VIF pruning.
