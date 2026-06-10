# Plan: Model Selection Experiment Runner

## Context

During the model development, 12+ regression models and 5 classification models were
manually run and metrics copied into a report. 
We want to make this experiement process less painful and more structured. In this plan, we want to design an automation that: defines model candidates with hyperparameter grids, run all combinations in one command, and have
every result logged to MLflow so we can sort by R² or weighted F1 and pick the winner.

This is a **developer exploration tool** — separate from the production training path
(`main.py`, `model_registry.py`). It lives in `backend/experiments/`, a sibling to `app/`,
with a one-way dependency (experiments → app, never app → experiments).

---

## Key Design Decisions & Tradeoffs

### 1. Pipeline factory functions (additive) vs duplication in experiments/

**Decision: Add `build_regression_pipeline(estimator)` and `build_classification_pipeline(estimator)`
factory functions to the existing `app/pipelines/` files.**

Reasons: The preprocessing steps (SCHEMA_SPECIFIC_BASE_STEPS, FeatureProcessingPipeline) must
be identical to production — if they diverge, experiment results won't transfer. Duplicating
pipeline construction in `experiments/` means any future preprocessing change needs to be made
twice. The factory is purely additive: the existing `PIPELINE_REGRESSION` constant is untouched.

### 2. experiments/ location: sibling to app/ (not inside app/)

**Decision: `backend/experiments/` — sibling to `app/`.**

Reasons: Experiments are a developer tool, not application code. Placing them inside `app/`
risks accidental import by FastAPI startup or model_registry. A sibling directory makes the
boundary explicit and keeps `pyproject.toml` clean (no new package entry).

### 3. Hyperparameter search: ParameterGrid (exhaustive)

**Decision: `sklearn.model_selection.ParameterGrid` — exhaustive, deterministic.**

Design Notes: With 12 model families × 2-4 param values per key, the grid is bounded (~60-80
regression runs, ~20 classification runs). ParameterGrid gives us every combination in a
predictable, reproducible way. RandomizedSearchCV adds sampling randomness without benefit at
this scale. Optuna/Bayesian would require a new heavy dependency for marginal gain.

### 4. Dataset split: full train.csv → fit, test.csv → evaluate

**Decision: Train on full `train.csv` (no holdout split), evaluate on separate `test.csv`.**

- test.csv has labels. Classification has severe class imbalance
(Duplex=36, Studio=5) — a random 80/20 holdout would leave only ~7 Duplex samples for
training, producing unreliable results. Training on the full labeled dataset gives the same
conditions as the report and produces metrics on the actual test set.

Optional `--val-split 0.2` flag retained for observability/learning-curve insight only,
not as the primary evaluation metric.

### 5. Flat MLflow runs (not nested), separate experiment names

**Decision: All runs are flat (no parent/child nesting) inside separate search experiments.**

- Nested runs require expanding each parent to sort across model families — you can't
sort all XGBoost variants against all LightGBM variants in one view. Flat runs with a
`model_family` tag are sortable by any metric in the MLflow UI table. Separate experiment names
(`house-price-regression-search`, `house-type-classification-search`) keep the production
experiments (`house-price-regression`, `house-type-classification`) clean.

### 6. MLflow: use directly, not via MLflowTracker

**Decision: Call `mlflow.*` directly in `experiments/runner.py` — do not reuse MLflowTracker.**

Rationale: `MLflowTracker` is write-only, coupled to `Settings`, and hard-codes the production
experiment name. The runner needs custom experiment names, `model_family` tags, and optional
artifact logging. Writing ~30 lines of direct MLflow calls is simpler than extending
MLflowTracker for a use case it was never designed for.

### 7. sample_weight for classification: computed per split, not pre-computed

**Decision: Compute `sample_weights` inside `_fit_and_log()` from the training fold.**

Reason: The production `classification_main()` computes weights from the full training set.
When the runner uses the full `train.csv`, the same weights apply. The weighting logic
(ratio < 0.01 → 10×, ratio < 0.05 → 5×, ratio < 0.1 → 3×, else 1×) is extracted into a
shared `_compute_sample_weights(y_encoded)` utility to avoid duplication between runner and
the existing classification_main.

### 8. Artifact logging: opt-in (off by default for speed)

**Decision: `--log-artifacts` flag, default False.**

Rationale: Logging 12 model families × sklearn pipeline (~50-300MB each) would consume
significant disk and slow each run by 5-30s. During search you care about metrics, not the
artifact. When you've identified the winner, run `main.py` (production path) to train and save
the final artifact properly.

---

## File Structure

```
backend/
├── app/
│   └── pipelines/
│       ├── regression.py       MODIFY — add build_regression_pipeline(estimator)
│       ├── classification.py   MODIFY — add build_classification_pipeline(estimator)
│       └── __init__.py         MODIFY — export new factories
│
└── experiments/                NEW — developer exploration tool
    ├── __init__.py             empty
    ├── candidates/
    │   ├── __init__.py
    │   ├── regression.py       12 RegressionCandidate entries with param grids
    │   └── classification.py   5 ClassificationCandidate entries with param grids
    ├── metrics.py              evaluate_regression(), evaluate_classification()
    ├── runner.py               RegressionRunner, ClassificationRunner classes
    ├── run_regression.py       CLI: python -m experiments.run_regression
    └── run_classification.py   CLI: python -m experiments.run_classification
```

---

## Core Data Structures

### `experiments/candidates/regression.py`

```python
@dataclass
class RegressionCandidate:
    family: str                  # display name and MLflow tag
    estimator: BaseEstimator     # base model (no TransformedTargetRegressor yet)
    param_grid: list[dict]       # list of dicts for ParameterGrid

    def expand(self) -> list[tuple[str, BaseEstimator, dict]]:
        """Returns (family, cloned+configured estimator, params) per grid entry."""
```

12 candidates matching the report: LightGBM, XGBoost, HistGBM, GradientBoosting,
RandomForest, ExtraTrees, AdaBoost, Ridge, Lasso, ElasticNet, VotingEnsemble, Stacking.

### `experiments/candidates/classification.py`

```python
@dataclass
class ClassificationCandidate:
    family: str
    estimator: ClassifierMixin
    param_grid: list[dict]

    def expand(self) -> list[tuple[str, ClassifierMixin, dict]]:
```

5 candidates: XGBoost, LightGBM, RandomForest, SVM, LogisticRegression.

### `experiments/metrics.py`

```python
def evaluate_regression(y_true, y_pred) -> RegressionMetrics
    # r2, mae, mse, rmse, explained_variance, max_error, median_absolute_error

def evaluate_classification(y_true, y_pred, classes) -> ClassificationMetrics
    # accuracy, weighted_f1, macro_f1, precision, recall, cohen_kappa, mcc, per_class_f1
```

Both return dataclasses with `.as_dict()` for MLflow logging.

---

## Pipeline Factories (additive change to existing files)

### `app/pipelines/regression.py` — add at bottom

```python
def build_regression_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Same preprocessing as PIPELINE_REGRESSION, swappable final estimator.
    Estimator is wrapped in TransformedTargetRegressor with log1p target transform.
    All regression candidates go through this — linear models especially benefit
    from the log transform on the right-skewed price target."""
    return Pipeline(steps=[
        *SCHEMA_SPECIFIC_BASE_STEPS,
        ("custom_features", CustomRegressionFeatures(sell_month_col="sell_month")),
        ("feature_processing", FeatureProcessingPipeline(...)),  # identical to constant
        ("regression", TransformedTargetRegressor(
            regressor=estimator,
            transformer=FunctionTransformer(np.log1p, inverse_func=np.expm1, ...),
        )),
    ])
```

**Note:** The existing `PIPELINE_REGRESSION` constant is left exactly as written. The factory
is a new addition that does not replace or refactor the constant.

### `app/pipelines/classification.py` — add at bottom

```python
def build_classification_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Step name is always 'classify' so classify__sample_weight routes correctly
    via sklearn's fit_params passthrough, matching the production pipeline."""
```

---

## Runner Core Logic

### `experiments/runner.py` — `RegressionRunner`

```python
class RegressionRunner:
    def __init__(self, experiment_name, tracking_uri, log_artifacts, random_state): ...

    def run(self, train_df, test_df, candidates) -> list[dict]:
        # split_x_y(train_df, "price", also_pop=["id"]) for train
        # split_x_y(test_df, "price") for evaluation
        # loop candidates → _fit_and_log → sort by r2 → print leaderboard

    def _fit_and_log(self, family, estimator, params, X_tr, y_tr, X_te, y_te) -> dict:
        pipeline = build_regression_pipeline(estimator)
        # time the fit
        # evaluate_regression(y_te, pipeline.predict(X_te))
        # mlflow.start_run(run_name=f"{family}_{params_hash(params)}")
        # mlflow.set_tags(model_family=family, params_hash=..., split="full_train+test")
        # mlflow.log_params({model_family, train_rows, test_rows, hp_*})
        # mlflow.log_metrics({r2, mae, rmse, ..., fit_time_s})
        # optionally: mlflow.sklearn.log_model(pipeline, ...)
```

### `experiments/runner.py` — `ClassificationRunner`

Same structure, with two differences:
1. Uses `LabelEncoder` on `type` column; passes `le.classes_` to `evaluate_classification`
2. Calls `pipeline.fit(X_tr, y=y_tr, classify__sample_weight=_compute_sample_weights(y_tr))`
3. Uses `stratify=y_tr` if val-split is requested (preserves class ratios in split)
4. Logs `label_classes` JSON tag for traceability

```python
def _compute_sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    # Identical logic to production classification_main:
    # ratio < 0.01 → 10×, ratio < 0.05 → 5×, ratio < 0.1 → 3×, else 1×
```

---

## CLI Entry Points

### `run_regression.py`
```bash
cd backend/
python -m experiments.run_regression \
    --train-data ../data/train.csv \
    --test-data  ../data/test.csv \
    --experiment-name house-price-regression-search \
    [--log-artifacts] \
    -vv
```

### `run_classification.py`
```bash
python -m experiments.run_classification \
    --train-data ../data/train.csv \
    --test-data  ../data/test.csv \
    --experiment-name house-type-classification-search \
    -vv
```

Both CLIs:
- `--train-data PATH` (required)
- `--test-data PATH` (required)
- `--experiment-name STR` (default shown above)
- `--tracking-uri STR` (default: `sqlite:///mlflow.db` in backend/)
- `--log-artifacts` (flag, default off)
- `--random-state INT` (default 42)
- `-v / -vv` (logging verbosity)

---

## MLflow UI Experience

After running both CLIs:

```
Experiment: house-price-regression-search
┌────────────────────────────┬──────┬────────┬────────┬────────────┐
│ Run name                   │  R²  │  MAE   │  RMSE  │ fit_time_s │
├────────────────────────────┼──────┼────────┼────────┼────────────┤
│ VotingEnsemble_a3f1...     │ 0.77 │ 281K   │ 603K   │ 8.6        │
│ HistGBM_b2c4...            │ 0.76 │ 291K   │ 622K   │ 7.1        │
│ LightGBM_n200_d8_lr008     │ 0.75 │ 295K   │ 637K   │ 1.0        │
│ ...                        │      │        │        │            │
└────────────────────────────┴──────┴────────┴────────┴────────────┘
Filter: tags.model_family = "LightGBM" → shows all LightGBM hyperparameter variants
Sort by: r2 desc → best model at top
```

---

## Implementation Sequence

1. Add factory functions to `app/pipelines/regression.py` and `app/pipelines/classification.py`
   — purely additive, existing constants untouched. Update `app/pipelines/__init__.py` exports.

2. Create `experiments/__init__.py`, `experiments/metrics.py`
   — no dependencies on candidates or runner; independently testable.

3. Create `experiments/candidates/regression.py`, `experiments/candidates/classification.py`
   — verify `ParameterGrid` expansion counts before wiring to runner.

4. Create `experiments/runner.py`
   — write `RegressionRunner` first (simpler, no sample_weight), test with 1 candidate + toy
   data, then add `ClassificationRunner`.

5. Create `experiments/run_regression.py`, `experiments/run_classification.py` CLI entry points.

---

## Known Edge Cases

| Issue | Candidate | Resolution |
|---|---|---|
| `VotingRegressor` is itself an ensemble — wrapping in TTR is correct | VotingEnsemble | TTR wraps the whole ensemble; log1p applies to target before ensemble sees it. Fine. |
| `StackingRegressor` with `cv=3` requires ≥3 samples per class | Stacking | Full train.csv has 5016 rows; no issue. |
| `SVC` slow on large data | SVM (classification) | 5016 rows, trains in <2min. Not a concern at this scale. |
| `AdaBoost` uses `DecisionTreeRegressor` as base, operates in log-space after TTR | AdaBoost | Converges correctly; metrics are in original price scale after inverse transform. |
| Classification val-split loses rare classes (Duplex=36, Studio=5) | all classifiers | Confirmed: train on full train.csv, evaluate on test.csv. No holdout split for classification. |

---

## Verification

```bash
cd backend/

# 1. Run regression search (~10-15 min for all 12 families)
python -m experiments.run_regression \
    --train-data ../data/train.csv \
    --test-data ../data/test.csv \
    -vv

# 2. Run classification search (~5 min for 5 families)
python -m experiments.run_classification \
    --train-data ../data/train.csv \
    --test-data ../data/test.csv \
    -vv

# 3. Open MLflow UI
mlflow ui \
    --backend-store-uri "sqlite:////$(pwd)/mlflow.db" \
    --port 5001
# → http://localhost:5001
# → Select "house-price-regression-search" → sort by r2 desc

# 4. Pick the winner → update PIPELINE_REGRESSION in app/pipelines/regression.py
#    → run main.py to produce the final production model
```

---

## Branch
`feat/experiment-runner`
