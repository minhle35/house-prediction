"""Experiment runner: fits every candidate in a grid and logs results to MLflow."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.pipelines import build_classification_pipeline, build_regression_pipeline
from experiments.metrics import evaluate_classification, evaluate_regression

if TYPE_CHECKING:
    from experiments.candidates.classification import ClassificationCandidate
    from experiments.candidates.regression import RegressionCandidate

log = logging.getLogger(__name__)


def _params_hash(params: dict) -> str:
    serialized = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def _compute_sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights matching production classification_main logic."""
    classes, counts = np.unique(y_encoded, return_counts=True)
    total = len(y_encoded)
    weights = np.ones(len(y_encoded), dtype=float)
    for cls, count in zip(classes, counts):
        ratio = count / total
        if ratio < 0.01:
            w = 10.0
        elif ratio < 0.05:
            w = 5.0
        elif ratio < 0.1:
            w = 3.0
        else:
            w = 1.0
        weights[y_encoded == cls] = w
    return weights


def _split_xy(
    df: pd.DataFrame, target: str, drop_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    extra = drop_cols or []
    y = df.pop(target)
    for col in extra:
        if col in df.columns:
            df.pop(col)
    return df, y


class RegressionRunner:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        log_artifacts: bool = False,
        random_state: int = 42,
        config_path: Path | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._log_artifacts = log_artifacts
        self._random_state = random_state
        self._config_path = config_path

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        log.info(
            "Regression runner — experiment: %s  uri: %s", experiment_name, tracking_uri
        )

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        candidates: list[RegressionCandidate],
    ) -> list[dict]:
        X_tr, y_tr = _split_xy(train_df, "price", drop_cols=["id"])
        X_te, y_te = _split_xy(test_df, "price", drop_cols=["id"])

        results: list[dict] = []
        total_grid_points = sum(len(list(c.expand())) for c in candidates)
        log.info(
            "Running %d grid points across %d families",
            total_grid_points,
            len(candidates),
        )

        for candidate in candidates:
            for family, estimator, params in candidate.expand():
                row = self._fit_and_log(
                    family, estimator, params, X_tr, y_tr, X_te, y_te
                )
                results.append(row)

        results.sort(key=lambda r: r["r2"], reverse=True)
        self._print_leaderboard(results)
        return results

    def _fit_and_log(
        self,
        family: str,
        estimator,
        params: dict,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_te: pd.DataFrame,
        y_te: pd.Series,
    ) -> dict:
        pipeline = build_regression_pipeline(estimator)

        t0 = time.perf_counter()
        try:
            pipeline.fit(X_tr, y_tr)
        except Exception:
            log.exception("Training failed for %s params=%s", family, params)
            return {
                "family": family,
                "params": params,
                "r2": float("-inf"),
                "error": True,
            }
        fit_time = time.perf_counter() - t0

        y_pred_raw = pipeline.predict(X_te)
        y_pred = None
        if isinstance(y_pred_raw, tuple):
            y_pred = y_pred_raw[0]
        else:
            y_pred = y_pred_raw
        m = evaluate_regression(np.asarray(y_te), y_pred)

        run_name = f"{family}_{_params_hash(params)}" if params else family
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"model_family": family, "split": "full_train+test"})
            mlflow.log_params(
                {
                    "model_family": family,
                    "train_rows": len(X_tr),
                    "test_rows": len(X_te),
                    **{f"hp_{k}": v for k, v in params.items()},
                }
            )
            mlflow.log_metrics({**m.as_dict(), "fit_time_s": fit_time})

            if self._config_path and self._config_path.exists():
                mlflow.log_artifact(str(self._config_path), artifact_path="config")

            if self._log_artifacts:
                mlflow.sklearn.log_model(pipeline, name=f"{family}_model")

        log.info(
            "%-20s r2=%.4f mae=%.0f fit=%.1fs params=%s",
            family,
            m.r2,
            m.mae,
            fit_time,
            params or "{}",
        )
        return {
            "family": family,
            "params": params,
            "r2": m.r2,
            "mae": m.mae,
            "rmse": m.rmse,
            "fit_time_s": fit_time,
        }

    @staticmethod
    def _print_leaderboard(results: list[dict]) -> None:
        print(
            f"\n{'Rank':<5} {'Family':<22} {'R²':>7} {'MAE':>12} {'RMSE':>12} {'Fit(s)':>8}"
        )
        print("-" * 72)
        for i, r in enumerate(results[:20], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.get("params", {}).items())
            name = f"{r['family']}({params_str})" if params_str else r["family"]
            print(
                f"{i:<5} {name:<22} {r['r2']:>7.4f} {r['mae']:>12,.0f} {r['rmse']:>12,.0f} {r['fit_time_s']:>8.1f}"
            )
        print()


class ClassificationRunner:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        log_artifacts: bool = False,
        random_state: int = 42,
        config_path: Path | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._log_artifacts = log_artifacts
        self._random_state = random_state
        self._config_path = config_path

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        log.info(
            "Classification runner — experiment: %s  uri: %s",
            experiment_name,
            tracking_uri,
        )

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        candidates: list[ClassificationCandidate],
    ) -> list[dict]:
        le = LabelEncoder()

        X_tr, y_tr_raw = _split_xy(train_df, "type", drop_cols=["id"])
        X_te, y_te_raw = _split_xy(test_df, "type", drop_cols=["id"])

        y_tr = np.asarray(le.fit_transform(y_tr_raw))
        y_te = np.asarray(le.transform(y_te_raw))
        sample_weights = _compute_sample_weights(y_tr)

        results: list[dict] = []
        total_grid_points = sum(len(list(c.expand())) for c in candidates)
        log.info(
            "Running %d grid points across %d families",
            total_grid_points,
            len(candidates),
        )

        for candidate in candidates:
            for family, estimator, params in candidate.expand():
                row = self._fit_and_log(
                    family,
                    estimator,
                    params,
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    le,
                    sample_weights,
                )
                results.append(row)

        results.sort(key=lambda r: r["weighted_f1"], reverse=True)
        self._print_leaderboard(results)
        return results

    def _fit_and_log(
        self,
        family: str,
        estimator,
        params: dict,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_te: pd.DataFrame,
        y_te: np.ndarray,
        le: LabelEncoder,
        sample_weights: np.ndarray,
    ) -> dict:
        pipeline = build_classification_pipeline(estimator)

        t0 = time.perf_counter()
        try:
            pipeline.fit(X_tr, y_tr, classify__sample_weight=sample_weights)
        except TypeError:
            # Some estimators (e.g. SVC) don't expose sample_weight via Pipeline.fit kwargs
            try:
                pipeline.fit(X_tr, y_tr)
            except Exception:
                log.exception("Training failed for %s params=%s", family, params)
                return {
                    "family": family,
                    "params": params,
                    "weighted_f1": float("-inf"),
                    "error": True,
                }
        except Exception:
            log.exception("Training failed for %s params=%s", family, params)
            return {
                "family": family,
                "params": params,
                "weighted_f1": float("-inf"),
                "error": True,
            }
        fit_time = time.perf_counter() - t0

        y_pred_raw = pipeline.predict(X_te)
        y_pred = None
        if isinstance(y_pred_raw, tuple):
            y_pred = y_pred_raw[0]
        else:
            y_pred = y_pred_raw
        m = evaluate_classification(y_te, y_pred, le.classes_)

        run_name = f"{family}_{_params_hash(params)}" if params else family
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "model_family": family,
                    "split": "full_train+test",
                    "label_classes": json.dumps(le.classes_.tolist()),
                }
            )
            mlflow.log_params(
                {
                    "model_family": family,
                    "train_rows": len(X_tr),
                    "test_rows": len(X_te),
                    **{f"hp_{k}": v for k, v in params.items()},
                }
            )
            mlflow.log_metrics({**m.as_dict(), "fit_time_s": fit_time})

            if self._config_path and self._config_path.exists():
                mlflow.log_artifact(str(self._config_path), artifact_path="config")

            if self._log_artifacts:
                mlflow.sklearn.log_model(pipeline, name=f"{family}_model")

        log.info(
            "%-20s wf1=%.4f acc=%.4f fit=%.1fs params=%s",
            family,
            m.weighted_f1,
            m.accuracy,
            fit_time,
            params or "{}",
        )
        return {
            "family": family,
            "params": params,
            "weighted_f1": m.weighted_f1,
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
            "fit_time_s": fit_time,
        }

    @staticmethod
    def _print_leaderboard(results: list[dict]) -> None:
        print(
            f"\n{'Rank':<5} {'Family':<22} {'wF1':>7} {'Acc':>7} {'macF1':>7} {'Fit(s)':>8}"
        )
        print("-" * 60)
        for i, r in enumerate(results[:20], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.get("params", {}).items())
            name = f"{r['family']}({params_str})" if params_str else r["family"]
            print(
                f"{i:<5} {name:<22} {r['weighted_f1']:>7.4f} {r['accuracy']:>7.4f} "
                f"{r['macro_f1']:>7.4f} {r['fit_time_s']:>8.1f}"
            )
        print()
