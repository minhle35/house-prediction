# Bug Report: Stateful Transformers Implemented as Stateless in sklearn Pipelines

---

## Root Cause

The sklearn `Pipeline` contract requires intermediate steps to implement both `fit` and `transform`:

| Method | When called | Responsibility |
| --- | --- | --- |
| `fit(X_train)` | Training only | Learn data-dependent parameters, store on `self` |
| `transform(X_any)` | Training and inference | Apply stored parameters — no learning |
| `fit_transform(X_train)` | Training only | Shorthand for `fit` then `transform` |
| `fit_predict(X_train)` | Training only | Shorthand for `fit` then `predict` |
| `predict(X_any)` | Training and inference | Apply fitted model — no learning |

**Key rule:** any method prefixed with `fit_` must never run on inference data.

Both bugs below share the same root cause: the class inherited the no-op `fit()` from `BaseTransformer` without overriding it, so data-dependent computation that belongs in `fit()` was placed inside `transform()` instead.

---

## Impacted Pipelines

```
SCHEMA_SPECIFIC_BASE_STEPS        ← backend/app/pipelines/base.py (shared)
  └── TransformDate               ← Bug 2 (both pipelines)

PIPELINE_REGRESSION               ← backend/app/pipelines/regression.py
  └── CustomRegressionFeatures    ← Bug 1 (regression only)

PIPELINE_CLASSIFICATION           ← backend/app/pipelines/classification.py
  └── PropertyTypeFeatures        ← OK (stateless, deterministic)
```

---

## Bug 1 — `CustomRegressionFeatures`: KMeans and StandardScaler fit inside `transform()`

**File:** `backend/app/transformer/feature_engineering.py`
**Pipeline:** `PIPELINE_REGRESSION` only

### Erroneous code

```python
@dataclass
class CustomRegressionFeatures(BaseTransformer):
    sell_month_col: str | None = None

    # No fit() override — inherits BaseTransformer.fit() which is a no-op

    def transform(self, X: pd.DataFrame) -> Any:
        clustering_df = X[clustering_features].dropna()
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(clustering_df)   # ← fits on whatever data is passed
        kmeans = KMeans(n_clusters=20, random_state=42)
        clusters = kmeans.fit_predict(scaled)          # ← centroids learned from same data
```

`scaler` and `kmeans` are local variables — created, used, and discarded on every call:

```
pipeline.fit(X_train):
    CustomRegressionFeatures.fit(X_train)       ← no-op
    CustomRegressionFeatures.transform(X_train) ← fits scaler+KMeans, discards them

pipeline.predict(X_new):
    CustomRegressionFeatures.transform(X_new)   ← fits NEW scaler+KMeans on test data, discards them
```

`postcode_cluster=3` at training time and `postcode_cluster=3` at inference time refer to different geographic groups — LightGBM's learned association between cluster IDs and prices is invalidated.

| Label | Accurate? |
| --- | --- |
| Data leakage | No — test data does not flow into LightGBM weights |
| Train-test inconsistency | Yes |
| Stateful transformer misuse | Yes |

### Fix

Promote clustering config to module-level constants, store `_scaler` and `_kmeans` as instance state, move fitting into `fit()`:

```python
_CLUSTERING_FEATURES = ["suburb_lat", "suburb_lng", "log_suburb_median_house_price", "log_km_from_cbd"]
N_CLUSTER = 20
RANDOM_STATE = 42

@dataclass
class CustomRegressionFeatures(BaseTransformer):
    sell_month_col: str | None = None
    _scaler: preprocessing.StandardScaler | None = field(default=None, init=False)
    _kmeans: KMeans | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: Any = None) -> Any:
        clustering_df = X[_CLUSTERING_FEATURES].dropna()
        self._scaler = preprocessing.StandardScaler()
        scaled = self._scaler.fit_transform(clustering_df)
        self._kmeans = KMeans(n_clusters=N_CLUSTER, random_state=RANDOM_STATE)
        self._kmeans.fit(scaled)
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        if self._scaler is None or self._kmeans is None:
            raise NotFittedError("CustomRegressionFeatures is not fitted yet.")

        clustering_df = X[_CLUSTERING_FEATURES].dropna()
        scaled = self._scaler.transform(clustering_df)   # reuse training-derived params
        clusters = self._kmeans.predict(scaled)          # reuse training-derived centroids
        X.loc[clustering_df.index, "postcode_cluster"] = clusters
        X["postcode_cluster"] = X["postcode_cluster"].fillna(-1).astype(int).astype(str)
        X["region_postcode_cluster"] = X["region"].astype(str) + "_" + X["postcode_cluster"]
        return X
```

---

## Bug 2 — `TransformDate`: `days_since_relative_to_min` computed from current data

**File:** `backend/app/transformer/preprocessing.py`
**Pipeline:** Both pipelines via `SCHEMA_SPECIFIC_BASE_STEPS`

### Erroneous code

```python
if self.days_since_relative_to_min:
    X[self.days_since_relative_to_min] = (
        X[self.target] - X[self.target].min()   # ← min() recomputed from current X
    ).dt.days
```

`X[self.target].min()` is the earliest sale date in **whatever batch is currently being transformed** — a learned statistic that must come from training data only:

```
pipeline.fit(X_train):
    min_date = X_train["date_sold"].min()   ← e.g. 2020-01-05
    sell_days_since_min = days since 2020-01-05

pipeline.predict(X_new):
    min_date = X_new["date_sold"].min()     ← e.g. 2022-06-12 (different anchor)
    sell_days_since_min = days since 2022-06-12
```

The worst case is a FastAPI server receiving one property at a time: `X_new.min()` equals the row's own date, so `sell_days_since_min=0` for every prediction.

| Label | Accurate? |
| --- | --- |
| Data leakage | No — test min date does not flow into model weights |
| Train-test inconsistency | Yes |
| Stateful transformer misuse | Yes |
| Silent failure at inference | Yes — no error raised, wrong values produced |

### Fix

Store the minimum date in `fit()`, reuse it in `transform()`:

```python
@dataclass
class TransformDate(BaseTransformer):
    # ... existing fields ...
    _min_date: pd.Timestamp | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y: Any = None) -> Any:
        if self.days_since_relative_to_min:
            dates = pd.to_datetime(X[self.target], format=self.target_datefmt)
            self._min_date = dates.min()
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        # ...
        if self.days_since_relative_to_min:
            if self._min_date is None:
                raise NotFittedError("TransformDate is not fitted yet.")
            X[self.days_since_relative_to_min] = (
                X[self.target] - self._min_date   # reuse training-derived anchor
            ).dt.days
```

---

## Summary

| Bug | Class | Statistic incorrectly in `transform()` | Pipelines affected |
| --- | --- | --- | --- |
| 1 | `CustomRegressionFeatures` | `StandardScaler` params, KMeans centroids | `PIPELINE_REGRESSION` |
| 2 | `TransformDate` | Minimum sale date (`X[target].min()`) | Both pipelines |
