import copy
import json
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.core.config import settings
from app.io.artifacts import ModelArtifacts
from app.io.tracking import ClassificationResult, RegressionResult, make_tracker
from app.pipelines import PIPELINE_CLASSIFICATION, PIPELINE_REGRESSION
from app.training.utils import split_x_y

log = logging.getLogger(__name__)

_artifacts = ModelArtifacts(settings)
_tracker = make_tracker(settings)

_REGRESSION_NAME = settings.regression_model_name
_CLASSIFICATION_NAME = settings.classification_model_name
_LABEL_ENCODER_NAME = "label_encoder"


@dataclass
class _ModelState:
    regression: Pipeline | None = field(default=None)
    classification: Pipeline | None = field(default=None)
    label_encoder: LabelEncoder | None = field(default=None)


_state = _ModelState()


def load_models() -> None:
    """Load models into memory.

    Load order:
    1. Try artifacts (Azure Blob → local joblib) for each of the three objects.
    2. If any are missing and train_data_path is configured, train from CSV and save.
    3. If still missing, raise RuntimeError so the app fails fast with a clear message.
    """
    _state.regression = _artifacts.load(_REGRESSION_NAME)
    _state.classification = _artifacts.load(_CLASSIFICATION_NAME)
    _state.label_encoder = _artifacts.load(_LABEL_ENCODER_NAME)

    missing = [
        name
        for name, obj in [
            (_REGRESSION_NAME, _state.regression),
            (_CLASSIFICATION_NAME, _state.classification),
            (_LABEL_ENCODER_NAME, _state.label_encoder),
        ]
        if obj is None
    ]

    if missing:
        log.info("Missing artifacts %s — attempting CSV training fallback", missing)
        _train_from_csv()

    missing_after = [
        name
        for name, obj in [
            (_REGRESSION_NAME, _state.regression),
            (_CLASSIFICATION_NAME, _state.classification),
            (_LABEL_ENCODER_NAME, _state.label_encoder),
        ]
        if obj is None
    ]
    if missing_after:
        raise RuntimeError(
            f"Models could not be loaded: {missing_after}. "
            "Set AZURE_STORAGE_CONNECTION_STRING or TRAIN_DATA_PATH."
        )

    log.info("All models loaded successfully")


def _train_from_csv() -> None:
    if not settings.train_data_path:
        log.warning("TRAIN_DATA_PATH not set — cannot train from CSV")
        return

    log.info("Training from CSV: %s", settings.train_data_path)
    train_df = pd.read_csv(settings.train_data_path)

    _fit_regression(train_df.copy())
    _fit_classification(train_df.copy())


def _fit_regression(train_df: pd.DataFrame) -> None:
    train_x, train_y = split_x_y(train_df, "price", also_pop=["id"])
    pipeline = copy.deepcopy(PIPELINE_REGRESSION)
    pipeline.fit(train_x, y=train_y)
    _state.regression = pipeline
    _artifacts.save(pipeline, _REGRESSION_NAME)
    log.info("Regression model trained and saved")

    # In-sample metrics (no separate test set available in this path)
    predicted = pipeline.predict(train_x)
    mse = float(sk_metrics.mean_squared_error(train_y, predicted))
    lgbm = pipeline.named_steps["regression"].regressor_
    fp = pipeline.named_steps["feature_processing"].get_params()
    params = {
        "n_estimators": lgbm.n_estimators,
        "max_depth": lgbm.max_depth,
        "learning_rate": lgbm.learning_rate,
        "random_state": lgbm.random_state,
        "correlation_threshold": fp["numerical"]["feature_correlation"]["threshold"],
        "variance_threshold": fp["numerical"]["feature_variance"]["threshold"],
        "knn_neighbors": fp["numerical"]["imputer"]["n_neighbors"],
        "train_rows": int(train_x.shape[0]),
        "train_cols": int(train_x.shape[1]),
        "eval_split": "train_only",
    }
    result = RegressionResult(
        r2=float(sk_metrics.r2_score(train_y, predicted)),
        mae=float(sk_metrics.mean_absolute_error(train_y, predicted)),
        mse=mse,
        rmse=math.sqrt(mse),
        explained_variance=float(sk_metrics.explained_variance_score(train_y, predicted)),
        max_error=float(sk_metrics.max_error(train_y, predicted)),
        median_absolute_error=float(sk_metrics.median_absolute_error(train_y, predicted)),
    )
    _tracker.log_regression(params, result, pipeline)


def _fit_classification(train_df: pd.DataFrame) -> None:
    train_x, train_y_raw = split_x_y(train_df, "type", also_pop=["id"])

    le = LabelEncoder()
    train_y: np.ndarray = le.fit_transform(train_y_raw)

    class_counts = pd.Series(train_y_raw).value_counts()
    majority = class_counts.max()
    class_weights: dict[int, float] = {
        int(le.transform([cls])[0]): (
            10.0 if (ratio := count / majority) < 0.01
            else 5.0 if ratio < 0.05
            else 3.0 if ratio < 0.1
            else 1.0
        )
        for cls, count in class_counts.items()
    }
    sample_weights = np.array([class_weights.get(c, 1.0) for c in train_y])

    pipeline = copy.deepcopy(PIPELINE_CLASSIFICATION)
    pipeline.fit(train_x, y=train_y, classify__sample_weight=sample_weights)

    _state.classification = pipeline
    _state.label_encoder = le
    _artifacts.save(pipeline, _CLASSIFICATION_NAME)
    _artifacts.save(le, _LABEL_ENCODER_NAME)
    log.info("Classification model trained and saved")

    # In-sample metrics
    predicted = pipeline.predict(train_x)
    f1_per_arr = sk_metrics.f1_score(train_y, predicted, average=None, zero_division=1)
    per_class_f1 = {str(cls): float(s) for cls, s in zip(le.classes_, f1_per_arr, strict=False)}
    fp = pipeline.named_steps["feature_processing"].get_params()
    params = {
        "correlation_threshold": fp["numerical"]["feature_correlation"]["threshold"],
        "variance_threshold": fp["numerical"]["feature_variance"]["threshold"],
        "train_rows": int(train_x.shape[0]),
        "num_classes": len(le.classes_),
        "class_weights": json.dumps({str(k): v for k, v in class_weights.items()}),
        "eval_split": "train_only",
    }
    result = ClassificationResult(
        accuracy=float(sk_metrics.accuracy_score(train_y, predicted)),
        weighted_f1=float(sk_metrics.f1_score(train_y, predicted, average="weighted", zero_division=1)),
        macro_f1=float(sk_metrics.f1_score(train_y, predicted, average="macro", zero_division=1)),
        precision=float(sk_metrics.precision_score(train_y, predicted, average="weighted", zero_division=1)),
        recall=float(sk_metrics.recall_score(train_y, predicted, average="weighted", zero_division=1)),
        cohen_kappa=float(sk_metrics.cohen_kappa_score(train_y, predicted)),
        matthews_corrcoef=float(sk_metrics.matthews_corrcoef(train_y, predicted)),
        per_class_f1=per_class_f1,
    )
    _tracker.log_classification(params, result, pipeline, le)


def get_regression() -> Pipeline:
    if _state.regression is None:
        raise RuntimeError("Regression model is not loaded")
    return _state.regression


def get_classification() -> tuple[Pipeline, LabelEncoder]:
    if _state.classification is None or _state.label_encoder is None:
        raise RuntimeError("Classification model or label encoder is not loaded")
    return _state.classification, _state.label_encoder
