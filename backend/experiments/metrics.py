import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

_INVALID_METRIC_CHARS = re.compile(r"[^a-zA-Z0-9_\-. :/]")


def _safe_metric_key(name: str) -> str:
    return _INVALID_METRIC_CHARS.sub("_", name)


@dataclass
class RegressionMetrics:
    r2: float
    mae: float
    mse: float
    rmse: float
    explained_variance: float
    max_error: float
    median_absolute_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "r2": self.r2,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "explained_variance": self.explained_variance,
            "max_error": self.max_error,
            "median_absolute_error": self.median_absolute_error,
        }


@dataclass
class ClassificationMetrics:
    accuracy: float
    weighted_f1: float
    macro_f1: float
    precision: float
    recall: float
    cohen_kappa: float
    matthews_corrcoef: float
    per_class_f1: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        base = {
            "accuracy": self.accuracy,
            "weighted_f1": self.weighted_f1,
            "macro_f1": self.macro_f1,
            "precision": self.precision,
            "recall": self.recall,
            "cohen_kappa": self.cohen_kappa,
            "matthews_corrcoef": self.matthews_corrcoef,
        }
        per_class = {
            f"f1_{_safe_metric_key(cls)}": score
            for cls, score in self.per_class_f1.items()
        }
        return {**base, **per_class}


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mse = mean_squared_error(y_true, y_pred)
    return RegressionMetrics(
        r2=r2_score(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        explained_variance=explained_variance_score(y_true, y_pred),
        max_error=max_error(y_true, y_pred),
        median_absolute_error=float(median_absolute_error(y_true, y_pred)),
    )


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> ClassificationMetrics:
    per_class_scores = np.atleast_1d(
        f1_score(y_true, y_pred, average=None, zero_division=0)
    )
    per_class_f1 = {
        str(cls): float(score) for cls, score in zip(classes, per_class_scores)
    }
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        precision=float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        cohen_kappa=cohen_kappa_score(y_true, y_pred),
        matthews_corrcoef=matthews_corrcoef(y_true, y_pred),
        per_class_f1=per_class_f1,
    )
