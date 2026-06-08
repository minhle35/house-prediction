"""Loads RegressionCandidate / ClassificationCandidate objects from YAML config files."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from experiments.candidates.regression import RegressionCandidate
from experiments.candidates.classification import ClassificationCandidate

_CANDIDATES_DIR = Path(__file__).parent


def _import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_estimator(class_path: str, params: dict[str, Any]):
    cls = _import_class(class_path)
    return cls(**params)


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _wrap_param_grid(param_grid_raw: dict[str, list]) -> list[dict]:
    """Wrap a YAML {param: [values]} dict into a list so ParameterGrid can expand it in expand()."""
    if not param_grid_raw:
        return [{}]
    return [param_grid_raw]


def load_regression_candidates(
    config_path: Path | None = None,
) -> list[RegressionCandidate]:
    path = config_path or _CANDIDATES_DIR / "regression.yaml"
    data = _load_yaml(path)
    candidates: list[RegressionCandidate] = []

    for entry in data["candidates"]:
        family = entry["family"]
        entry_type = entry.get("type", "standard")

        if entry_type == "voting":
            from sklearn.ensemble import VotingRegressor
            sub = [
                (e["name"], _build_estimator(e["class"], e.get("params", {})))
                for e in entry["estimators"]
            ]
            estimator = VotingRegressor(estimators=sub)
            candidates.append(RegressionCandidate(family=family, estimator=estimator, param_grid=[{}]))

        elif entry_type == "stacking":
            from sklearn.ensemble import StackingRegressor
            sub = [
                (e["name"], _build_estimator(e["class"], e.get("params", {})))
                for e in entry["estimators"]
            ]
            fe_cfg = entry.get("final_estimator", {})
            final = _build_estimator(fe_cfg["class"], fe_cfg.get("params", {})) if fe_cfg else None
            estimator = StackingRegressor(
                estimators=sub,
                final_estimator=final,
                cv=entry.get("cv", 5),
                n_jobs=-1,
            )
            candidates.append(RegressionCandidate(family=family, estimator=estimator, param_grid=[{}]))

        else:
            defaults = entry.get("defaults") or {}
            base_estimator = _build_estimator(entry["class"], defaults)
            raw_grid = entry.get("param_grid") or {}
            param_grid = _wrap_param_grid(raw_grid)
            candidates.append(
                RegressionCandidate(family=family, estimator=base_estimator, param_grid=param_grid)
            )

    return candidates


def load_classification_candidates(
    config_path: Path | None = None,
) -> list[ClassificationCandidate]:
    path = config_path or _CANDIDATES_DIR / "classification.yaml"
    data = _load_yaml(path)
    candidates: list[ClassificationCandidate] = []

    for entry in data["candidates"]:
        family = entry["family"]
        defaults = entry.get("defaults") or {}
        base_estimator = _build_estimator(entry["class"], defaults)
        raw_grid = entry.get("param_grid") or {}
        param_grid = _wrap_param_grid(raw_grid)
        candidates.append(
            ClassificationCandidate(family=family, estimator=base_estimator, param_grid=param_grid)
        )

    return candidates
