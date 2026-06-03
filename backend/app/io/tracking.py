import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.core.config import Settings

# Absolute path to backend/ so DB and mlruns/ are always in the same place
# regardless of the working directory when training is invoked.
_BACKEND_DIR = Path(__file__).parents[2]

log = logging.getLogger(__name__)

_INVALID_METRIC_CHARS = re.compile(r"[^a-zA-Z0-9_\-. :/]")


def _safe_metric_key(name: str) -> str:
    """Replace characters that MLflow metric names don't allow with underscores."""
    return _INVALID_METRIC_CHARS.sub("_", name)


@dataclass
class RegressionResult:
    r2: float
    mae: float
    mse: float
    rmse: float
    explained_variance: float
    max_error: float
    median_absolute_error: float


@dataclass
class ClassificationResult:
    accuracy: float
    weighted_f1: float
    macro_f1: float
    precision: float
    recall: float
    cohen_kappa: float
    matthews_corrcoef: float
    per_class_f1: dict[str, float] = field(default_factory=dict)


class MLflowTracker:
    """Logs params, metrics, and model artifacts to MLflow.

    Disabled by default (settings.mlflow_enabled=False).  When disabled every
    method is a no-op so callers never need to check the flag themselves.
    Model registration is threshold-gated and also off by default.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._enabled = settings.mlflow_enabled

        if not self._enabled:
            return

        import mlflow

        uri = settings.mlflow_tracking_uri or f"sqlite:///{_BACKEND_DIR}/mlflow.db"
        mlflow.set_tracking_uri(uri)

        log.info("MLflow tracking enabled — URI: %s", uri)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_regression(
        self,
        params: dict,
        result: RegressionResult,
        pipeline: Pipeline,
    ) -> str | None:
        """Log a regression training run. Returns the MLflow run ID, or None if disabled."""
        if not self._enabled:
            return None

        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment(self._settings.mlflow_experiment_regression)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"regression_{ts}") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "r2": result.r2,
                    "mae": result.mae,
                    "mse": result.mse,
                    "rmse": result.rmse,
                    "explained_variance": result.explained_variance,
                    "max_error": result.max_error,
                    "median_absolute_error": result.median_absolute_error,
                }
            )
            model_info = mlflow.sklearn.log_model(pipeline, name=f"regression_{ts}")

            if (
                self._settings.mlflow_register_models
                and result.r2 >= self._settings.mlflow_r2_threshold
            ):
                mlflow.register_model(model_info.model_uri, "house-price-regression")
                log.info(
                    "Registered regression model (r2=%.4f >= threshold %.2f)",
                    result.r2,
                    self._settings.mlflow_r2_threshold,
                )
            elif self._settings.mlflow_register_models:
                log.info(
                    "Regression model NOT registered (r2=%.4f < threshold %.2f)",
                    result.r2,
                    self._settings.mlflow_r2_threshold,
                )

            log.info(
                "MLflow regression run recorded — %s\n  model: %s",
                run.info.run_id,
                model_info.model_uri,
            )
            return run.info.run_id

    def log_classification(
        self,
        params: dict,
        result: ClassificationResult,
        pipeline: Pipeline,
        le: LabelEncoder,
    ) -> str | None:
        """Log a classification training run. Returns the MLflow run ID, or None if disabled."""
        if not self._enabled:
            return None

        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment(self._settings.mlflow_experiment_classification)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=f"classification_{ts}") as run:
            mlflow.log_params(params)

            scalar_metrics = {
                "accuracy": result.accuracy,
                "weighted_f1": result.weighted_f1,
                "macro_f1": result.macro_f1,
                "precision": result.precision,
                "recall": result.recall,
                "cohen_kappa": result.cohen_kappa,
                "matthews_corrcoef": result.matthews_corrcoef,
            }
            per_class = {
                f"f1_{_safe_metric_key(cls)}": score
                for cls, score in result.per_class_f1.items()
            }
            mlflow.log_metrics({**scalar_metrics, **per_class})

            model_info = mlflow.sklearn.log_model(pipeline, name=f"classification_{ts}")

            # Store label encoder classes as a tag for traceability
            mlflow.set_tag("label_classes", json.dumps(le.classes_.tolist()))

            if (
                self._settings.mlflow_register_models
                and result.weighted_f1 >= self._settings.mlflow_f1_threshold
            ):
                mlflow.register_model(model_info.model_uri, "house-type-classification")
                log.info(
                    "Registered classification model (weighted_f1=%.4f >= threshold %.2f)",
                    result.weighted_f1,
                    self._settings.mlflow_f1_threshold,
                )
            elif self._settings.mlflow_register_models:
                log.info(
                    "Classification model NOT registered (weighted_f1=%.4f < threshold %.2f)",
                    result.weighted_f1,
                    self._settings.mlflow_f1_threshold,
                )

            log.info(
                "MLflow classification run recorded — %s\n  model: %s",
                run.info.run_id,
                model_info.model_uri,
            )
            return run.info.run_id


class NullTracker(MLflowTracker):
    """Drop-in tracker that does nothing. Used when mlflow_enabled=False or in unit tests."""

    def __init__(self) -> None:
        # Bypass parent __init__ — no Settings needed, no MLflow imported
        self._enabled = False
        self._settings = None  # type: ignore[assignment]

    def log_regression(
        self, params: dict, result: RegressionResult, pipeline: Pipeline
    ) -> None:
        return None

    def log_classification(
        self,
        params: dict,
        result: ClassificationResult,
        pipeline: Pipeline,
        le: LabelEncoder,
    ) -> None:
        return None


def make_tracker(settings: Settings) -> MLflowTracker:
    """Factory: returns NullTracker when disabled, MLflowTracker when enabled."""
    if not settings.mlflow_enabled:
        return NullTracker()
    return MLflowTracker(settings)
