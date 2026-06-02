"""Unit tests for MLflowTracker and NullTracker.

No external MLflow server required — all tests use a tmp_path file-store URI.
"""

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.core.config import Settings
from app.io.tracking import (
    ClassificationResult,
    MLflowTracker,
    NullTracker,
    RegressionResult,
    make_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg_result(r2: float = 0.76) -> RegressionResult:
    return RegressionResult(
        r2=r2,
        mae=50000.0,
        mse=5e9,
        rmse=70710.0,
        explained_variance=0.77,
        max_error=300000.0,
        median_absolute_error=40000.0,
    )


def _cls_result(weighted_f1: float = 0.91) -> ClassificationResult:
    return ClassificationResult(
        accuracy=0.91,
        weighted_f1=weighted_f1,
        macro_f1=0.85,
        precision=0.92,
        recall=0.91,
        cohen_kappa=0.88,
        matthews_corrcoef=0.87,
        per_class_f1={"House": 0.95, "Apartment": 0.88, "Duplex": 0.70},
    )


def _dummy_regression_pipeline() -> Pipeline:
    return Pipeline([("model", LinearRegression())])


def _dummy_classification_pipeline() -> Pipeline:
    return Pipeline([("model", LogisticRegression(max_iter=10))])


def _label_encoder(*classes: str) -> LabelEncoder:
    import numpy as np

    le = LabelEncoder()
    le.fit(np.array(list(classes)))
    return le


def _tracker(
    tmp_path,
    *,
    register: bool = False,
    r2_threshold: float = 0.70,
    f1_threshold: float = 0.80,
) -> MLflowTracker:
    s = Settings(
        mlflow_enabled=True,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_register_models=register,
        mlflow_r2_threshold=r2_threshold,
        mlflow_f1_threshold=f1_threshold,
    )
    return MLflowTracker(s)


# ---------------------------------------------------------------------------
# NullTracker
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_null_tracker_returns_none_regression():
    nt = NullTracker()
    result = nt.log_regression({}, _reg_result(), _dummy_regression_pipeline())
    assert result is None


@pytest.mark.unit
def test_null_tracker_returns_none_classification():
    nt = NullTracker()
    le = _label_encoder("House", "Apartment")
    result = nt.log_classification(
        {}, _cls_result(), _dummy_classification_pipeline(), le
    )
    assert result is None


@pytest.mark.unit
def test_make_tracker_returns_null_when_disabled(tmp_path):
    s = Settings(mlflow_enabled=False)
    tracker = make_tracker(s)
    assert isinstance(tracker, NullTracker)


@pytest.mark.unit
def test_make_tracker_returns_mlflow_when_enabled(tmp_path):
    s = Settings(
        mlflow_enabled=True, mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"
    )
    tracker = make_tracker(s)
    assert isinstance(tracker, MLflowTracker)
    assert not isinstance(tracker, NullTracker)


@pytest.mark.unit
def test_null_tracker_writes_no_files(tmp_path):
    nt = NullTracker()
    nt.log_regression({}, _reg_result(), _dummy_regression_pipeline())
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# MLflowTracker — params and metrics logged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_regression_run_id_returned(tmp_path):
    tracker = _tracker(tmp_path)
    run_id = tracker.log_regression(
        {"n_estimators": 200}, _reg_result(), _dummy_regression_pipeline()
    )
    assert run_id is not None
    assert len(run_id) == 32


@pytest.mark.unit
def test_regression_params_logged(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path)
    params = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.08}
    run_id = tracker.log_regression(params, _reg_result(), _dummy_regression_pipeline())

    run = mlflow.get_run(run_id)
    assert run.data.params["n_estimators"] == "200"
    assert run.data.params["max_depth"] == "8"
    assert run.data.params["learning_rate"] == "0.08"


@pytest.mark.unit
def test_regression_metrics_logged(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path)
    result = _reg_result(r2=0.76)
    run_id = tracker.log_regression({}, result, _dummy_regression_pipeline())

    run = mlflow.get_run(run_id)
    assert run.data.metrics["r2"] == pytest.approx(0.76)
    assert "mae" in run.data.metrics
    assert "mse" in run.data.metrics
    assert "rmse" in run.data.metrics
    assert "explained_variance" in run.data.metrics
    assert "max_error" in run.data.metrics
    assert "median_absolute_error" in run.data.metrics


@pytest.mark.unit
def test_classification_run_id_returned(tmp_path):
    tracker = _tracker(tmp_path)
    le = _label_encoder("House", "Apartment", "Duplex")
    run_id = tracker.log_classification(
        {}, _cls_result(), _dummy_classification_pipeline(), le
    )
    assert run_id is not None


@pytest.mark.unit
def test_classification_metrics_logged(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path)
    le = _label_encoder("House", "Apartment", "Duplex")
    run_id = tracker.log_classification(
        {}, _cls_result(), _dummy_classification_pipeline(), le
    )

    run = mlflow.get_run(run_id)
    assert run.data.metrics["accuracy"] == pytest.approx(0.91)
    assert run.data.metrics["weighted_f1"] == pytest.approx(0.91)
    assert run.data.metrics["macro_f1"] == pytest.approx(0.85)
    assert "precision" in run.data.metrics
    assert "recall" in run.data.metrics
    assert "cohen_kappa" in run.data.metrics
    assert "matthews_corrcoef" in run.data.metrics


@pytest.mark.unit
def test_classification_per_class_f1_logged(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path)
    le = _label_encoder("House", "Apartment", "Duplex")
    run_id = tracker.log_classification(
        {}, _cls_result(), _dummy_classification_pipeline(), le
    )

    run = mlflow.get_run(run_id)
    assert run.data.metrics["f1_House"] == pytest.approx(0.95)
    assert run.data.metrics["f1_Apartment"] == pytest.approx(0.88)
    assert run.data.metrics["f1_Duplex"] == pytest.approx(0.70)


@pytest.mark.unit
def test_classification_label_classes_tag_logged(tmp_path):
    import json
    import mlflow

    tracker = _tracker(tmp_path)
    le = _label_encoder("Apartment", "Duplex", "House")
    run_id = tracker.log_classification(
        {}, _cls_result(), _dummy_classification_pipeline(), le
    )

    run = mlflow.get_run(run_id)
    classes = json.loads(run.data.tags["label_classes"])
    assert set(classes) == {"Apartment", "Duplex", "House"}


# ---------------------------------------------------------------------------
# Model registration — threshold gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registration_skips_below_r2_threshold(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path, register=True, r2_threshold=0.70)
    # r2=0.50 is below threshold → should NOT register
    run_id = tracker.log_regression(
        {}, _reg_result(r2=0.50), _dummy_regression_pipeline()
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    registered = client.search_registered_models()
    assert not any(m.name == "house-price-regression" for m in registered)


@pytest.mark.unit
def test_registration_fires_above_r2_threshold(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path, register=True, r2_threshold=0.70)
    # r2=0.80 is above threshold → should register
    tracker.log_regression({}, _reg_result(r2=0.80), _dummy_regression_pipeline())

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    registered = client.search_registered_models()
    assert any(m.name == "house-price-regression" for m in registered)


@pytest.mark.unit
def test_registration_skips_below_f1_threshold(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path, register=True, f1_threshold=0.80)
    le = _label_encoder("House", "Apartment")
    # weighted_f1=0.60 is below threshold → should NOT register
    tracker.log_classification(
        {}, _cls_result(weighted_f1=0.60), _dummy_classification_pipeline(), le
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    registered = client.search_registered_models()
    assert not any(m.name == "house-type-classification" for m in registered)


@pytest.mark.unit
def test_registration_fires_above_f1_threshold(tmp_path):
    import mlflow

    tracker = _tracker(tmp_path, register=True, f1_threshold=0.80)
    le = _label_encoder("House", "Apartment")
    # weighted_f1=0.91 is above threshold → should register
    tracker.log_classification(
        {}, _cls_result(weighted_f1=0.91), _dummy_classification_pipeline(), le
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    registered = client.search_registered_models()
    assert any(m.name == "house-type-classification" for m in registered)


@pytest.mark.unit
def test_registration_disabled_by_default(tmp_path):
    """register_models=False means no model is registered even when metrics pass."""
    import mlflow

    tracker = _tracker(tmp_path, register=False)
    tracker.log_regression({}, _reg_result(r2=0.99), _dummy_regression_pipeline())

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    assert client.search_registered_models() == []
