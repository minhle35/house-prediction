from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid


@dataclass
class RegressionCandidate:
    family: str
    estimator: BaseEstimator
    param_grid: list[dict[str, Any]] = field(default_factory=lambda: [{}])

    def expand(self) -> list[tuple[str, BaseEstimator, dict[str, Any]]]:
        """Return (family, configured-estimator-clone, params) for every grid point."""
        results = []
        for params in ParameterGrid(self.param_grid):
            est = clone(self.estimator)
            if params:
                est.set_params(**params)
            results.append((self.family, est, params))
        return results
