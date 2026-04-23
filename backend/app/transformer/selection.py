import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn import feature_selection
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from backend.app.transformer.base import BaseTransformer


log = logging.getLogger(__name__)


@dataclass
class CorrelationFeatureSelector(BaseEstimator):
    """Removes highly correlated numerical features.

    Parameters
    ----------
    cols_choose_from: list[str], optional
        The columns to choose from. When not set, will use all numerical columns
    threshold: float, optional
        The threshold correlation between features before dropping; once exceeded the column will be
        dropped. Default is 0.90.
    """

    cols_choose_from: list[str] | None = None
    threshold: float = 0.9

    _to_drop: list[str] | None = field(default=None, init=False)

    def fit(self, X: pd.DataFrame, y=None) -> Any:  # noqa: ANN001, ARG002
        if self.cols_choose_from is None:
            self.cols_choose_from = X.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

        corr_matrix = X[self.cols_choose_from].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self._to_drop = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] > self.threshold)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        if self._to_drop is None:
            raise NotFittedError("CorrelationFeatureSelector is not fitted yet.")

        log.info("CorrelationFeatureSelector dropping %s", self._to_drop)
        return X.drop(columns=self._to_drop)


@dataclass
class VarianceFeatureSelector(BaseTransformer):
    """Remove numerical features with variance below a threshold.

    Parameters
    ----------
    cols_choose_from: list[str], optional
        The columns to choose from. When not set, will use all numerical columns
    threshold: float, optional
        The variance threshold. Defaults to 0
    """

    cols_choose_from: list[str] | None = None
    threshold: float = 0

    _selector: feature_selection.VarianceThreshold | None = field(
        default=None, init=False
    )

    def fit(self, X: pd.DataFrame, y=None) -> Any:  # noqa: ANN001, ARG002
        if self.cols_choose_from is None:
            self.cols_choose_from = X.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

        self._selector = feature_selection.VarianceThreshold(self.threshold)
        self._selector.fit(X[self.cols_choose_from])
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        if self._selector is None:
            raise NotFittedError("VarianceFeatureSelector is not fitted yet.")
        mask = self._selector.get_support()

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "VarianceFeatureSelector (feature, mask, variance): %s",
                list(
                    zip(
                        self._selector.feature_names_in_,
                        mask,
                        self._selector.variances_,
                        strict=False,
                    )
                ),
            )

        drop = [
            col
            for idx, col in enumerate(self._selector.feature_names_in_)
            if not mask[idx]
        ]
        log.info("VarianceFeatureSelector dropping %s", drop)
        return X.drop(columns=drop)


@dataclass
class FeatureProcessingPipeline(BaseTransformer):
    """Feature processing pipeline, split for numerical and categorical steps.

    Parameters
    ----------
    numerical_steps: list[tuple[str, BaseEstimator]]
        Numerical processing steps, will only apply to `float`/`int` columns
    categorical_steps: list[tuple[str, BaseEstimator]], optional
        Categorical processing steps, will only apply to `object`/`category` columns. Will ignore if
        not set. Defaults to None (unset).
    exclude: list[str], optional
        Columns to exclude from the dataframe. Mutually exclusive to `include`.
    exclude: list[str], optional
        Columns to include from the dataframe. Mutually exclusive to `exclude`.
    """

    numerical_steps: list[tuple[str, BaseEstimator]]
    categorical_steps: list[tuple[str, BaseEstimator]] | None = None
    exclude: list[str] = field(default_factory=list)
    include: list[str] = field(default_factory=list)

    _composer: ColumnTransformer | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.exclude and self.include:
            raise ValueError("Can only set one of 'include' or 'exclude'")

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002, FBT001, FBT002
        categorical = (
            {
                "categorical": {
                    name: step.get_params() for name, step in self.categorical_steps
                }
            }
            if self.categorical_steps
            else {}
        )
        return {
            "numerical": {
                name: step.get_params() for name, step in self.numerical_steps
            },
            **categorical,
        }

    def fit(self, X: pd.DataFrame, y=None) -> Any:  # noqa: ANN001
        df = X[self.include] if self.include else X.drop(columns=self.exclude)

        numeric_features = df.select_dtypes(include=["float", "int"]).columns
        categorical_features = df.select_dtypes(include=["object", "category"]).columns

        if log.isEnabledFor(logging.DEBUG):
            orig_max_cols = pd.options.display.max_columns
            pd.options.display.max_columns = None  # pyright: ignore[reportAttributeAccessIssue]
            log.debug("Numerical features: %s", numeric_features)
            log.debug("Sample numerical values:\n%s", df[numeric_features].head(3))
            log.debug("Categorical features: %s", categorical_features)
            log.debug(
                "Sample categorical values:\n%s", df[categorical_features].head(3)
            )
            pd.options.display.max_columns = orig_max_cols

        transformers = [("num", Pipeline(steps=self.numerical_steps), numeric_features)]
        if self.categorical_steps:
            transformers.append(
                ("cat", Pipeline(steps=self.categorical_steps), categorical_features)
            )

        transformer = ColumnTransformer(transformers=transformers)
        transformer.fit(X, y=y)
        self._transformer = transformer
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        if self._transformer is None:
            raise NotFittedError("FeatureSelect is not fitted yet.")
        return self._transformer.transform(X)
