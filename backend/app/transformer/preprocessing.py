import pandas as pd
from dataclasses import dataclass
from typing import Any, Callable

from main import BaseTransformer


@dataclass
class TransformDate(BaseTransformer):
    """Transforms a target date column into separate date-related features.

    Parameters
    ----------
    target: str
        The target date column
    target_datefmt: str, optional
        The target column date format. Defaults to "%Y-%m-%d".
    datum: str, optional
        The datum date to use for calculating `days_since` column. Defaults to 'today'
    drop_target: bool, optional
        When set to true, will drop the target date column from the output. Defaults to True.
    days_since: str, optional
        When set, will output a column calculating the days since the datum. The value is the output
        column name.
    days_since_relative_to_min: str, optional
        When set, will output a column calculating the days since the minimum date in the column.
        The value is the output column name.
    year: str, optional
        When set, will output a column with name as the passed value, representing the year of the
        date.
    month: str, optional
        When set, will output a column with name as the passed value, representing the month of the
        date.
    season: str, optional
        When set, will output a column with name as the passed value, representing the season of the
        date.
    """

    target: str
    target_datefmt: str = "%Y-%m-%d"
    datum: str = "today"
    drop_target: bool = True

    days_since: str | None = None
    days_since_relative_to_min: str | None = None
    year: str | None = None
    month: str | None = None
    season: str | None = None

    def transform(self, X: pd.DataFrame) -> Any:
        datum = pd.Timestamp(self.datum)
        X[self.target] = pd.to_datetime(X[self.target], format=self.target_datefmt)

        if self.days_since:
            X[self.days_since] = (datum - X[self.target]).dt.days  # pyright: ignore[reportOperatorIssue]
        if self.days_since_relative_to_min:
            X[self.days_since_relative_to_min] = (
                X[self.target] - X[self.target].min()
            ).dt.days
        if self.year:
            X[self.year] = X[self.target].dt.year
        if self.month:
            X[self.month] = X[self.target].dt.month
        if self.season:
            X[self.season] = X[self.target].dt.month % 12 // 3 + 1

        if self.drop_target:
            return X.drop(columns=[self.target])
        return X


@dataclass
class PandasColumnTransform(BaseTransformer):
    """Applies a function to a target columns.

    Parameters
    ----------
    func: Callable
        The function to apply
    columns: list[str]
        The list of columns to apply over
    outputs: list[str], optional
        When set, must be same length as `columns`, and will be used as the output columns for the
        processed targets.
    """

    func: Callable
    columns: list[str]
    outputs: list[str] | None = None

    def __post_init__(self) -> None:
        if self.outputs is not None and len(self.outputs) != len(self.columns):
            raise ValueError("output must be same length as columns")

    def transform(self, X: pd.DataFrame) -> Any:
        output = self.outputs or self.columns
        X[output] = X[self.columns].apply(self.func)
        return X


@dataclass
class ZeroToNan(BaseTransformer):
    """Replaces zero-values with NaN for target columns.

    Parameters
    ----------
    columns: list[str]
        The columns to apply this transform for.
    """

    columns: list[str]

    def transform(self, X: pd.DataFrame) -> Any:
        X[self.columns] = X[self.columns].replace(0, np.nan)
        return X
