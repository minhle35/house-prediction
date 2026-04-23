from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from typing import Any


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base estimator / transformer class for use in sklearn.pipeline.Pipeline"""

    def fit(self, X, y=None) -> Any:  # noqa: ANN001, ARG002
        return self

    def transform(self, X) -> Any:  # noqa: ANN001
        return X
