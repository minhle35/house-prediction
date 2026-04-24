import logging
from pathlib import Path


import pandas as pd
import numpy.typing as npt
from sklearn import metrics
from sklearn.pipeline import Pipeline

from app.training.utils import pipeline_timed_context, split_x_y

log = logging.getLogger(__name__)


def regression_main(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output: Path,
) -> None:
    """Fit a regression model over training data and predict for test data.

    Parameters
    ----------
    train_df: pandas.Dataframe
        The dataframe to train over. This will be split with `split_x_y` with `y_col="price"`.
    test_df: pandas.Dataframe
        The dataframe to predict for. The y values are ignored here. Must have an ID column for the
        output.
    output: Path
        The output file to save the predictions over `test_df` to
    """

    train_x, train_y = split_x_y(train_df, "price", also_pop=["id"])
    test_x, test_y = split_x_y(test_df, "price")

    log.info("Regression fitting...")
    with pipeline_timed_context(pipeline) as pl:  # For debugging; use `-v`...
        pl.fit(train_x, y=train_y)

    log.info("Regression predicting...")
    predicted_y: npt.NDArray = pipeline.predict(test_x)  # type: ignore[return]
    final_df = test_x.assign(price=predicted_y.tolist())
    final_df[["id", "price"]].to_csv(output, index=False)

    if log.isEnabledFor(logging.INFO):  # For debugging; use `-v`...
        mm = {
            "MSE": metrics.mean_squared_error,
            "MAE": metrics.mean_absolute_error,
            "R2": metrics.r2_score,
            "Expl var": metrics.explained_variance_score,
            "Max err": metrics.max_error,
            "Med err": metrics.median_absolute_error,
        }
        for key, fn in mm.items():
            val = f"{fn(test_y, predicted_y):,.2f}"
            log.info("Regression %-10s: %s", key, val)
