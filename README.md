# House Prediction

Predicts Australian residential property **sale price** (regression) and **property type** (classification) from suburb and property features using scikit-learn pipelines.

## Models

| Task | Model | Target |
|---|---|---|
| Regression | LightGBM + log-transform | `price` |
| Classification | XGBoost + class weighting | `type` |

Both pipelines share a common preprocessing stage: zero→NaN imputation, date decomposition, log-scaling of skewed columns, and transport-time binning. Regression adds geo-clustering and cyclical month encoding; classification adds duplex-likelihood scoring.

## Usage

```bash
python main.py <train.csv> <test.csv> [-v] [-vv]
```

Outputs two CSV files:
- `regression.csv` — `id, price` predictions
- `classification.csv` — `id, type` predictions

| Flag | Effect |
|---|---|
| `-v` | INFO logging (step timings, class distribution, metrics) |
| `-vv` | DEBUG logging (feature names, sample values) |

## Pipeline steps

```
zero_to_nan → transform_date → log_cols → bin_transport_time
    → [custom features] → correlation filter → variance filter
    → impute → scale → encode → model
```

## Requirements

- Python 3.11+
- `lightgbm`, `xgboost`, `scikit-learn`, `pandas`, `numpy`
