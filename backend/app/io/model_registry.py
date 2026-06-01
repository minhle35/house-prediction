import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.core.config import settings
from app.io.artifacts import ModelArtifacts
from app.pipelines import PIPELINE_CLASSIFICATION, PIPELINE_REGRESSION
from app.training.utils import split_x_y

log = logging.getLogger(__name__)

_artifacts = ModelArtifacts(settings)

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


def _fit_classification(train_df: pd.DataFrame) -> None:
    train_x, train_y_raw = split_x_y(train_df, "type", also_pop=["id"])

    le = LabelEncoder()
    train_y: np.ndarray = le.fit_transform(train_y_raw)

    class_counts = pd.Series(train_y_raw).value_counts()
    majority = class_counts.max()
    weights = {
        le.transform([cls])[0]: (
            10.0 if (ratio := count / majority) < 0.01
            else 5.0 if ratio < 0.05
            else 3.0 if ratio < 0.1
            else 1.0
        )
        for cls, count in class_counts.items()
    }
    sample_weights = np.array([weights.get(c, 1.0) for c in train_y])

    pipeline = copy.deepcopy(PIPELINE_CLASSIFICATION)
    pipeline.fit(train_x, y=train_y, classify__sample_weight=sample_weights)

    _state.classification = pipeline
    _state.label_encoder = le
    _artifacts.save(pipeline, _CLASSIFICATION_NAME)
    _artifacts.save(le, _LABEL_ENCODER_NAME)
    log.info("Classification model trained and saved")


def get_regression() -> Pipeline:
    if _state.regression is None:
        raise RuntimeError("Regression model is not loaded")
    return _state.regression


def get_classification() -> tuple[Pipeline, LabelEncoder]:
    if _state.classification is None or _state.label_encoder is None:
        raise RuntimeError("Classification model or label encoder is not loaded")
    return _state.classification, _state.label_encoder
