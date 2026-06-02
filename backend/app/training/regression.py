import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import numpy.typing as npt
from sklearn import metrics
from sklearn.pipeline import Pipeline

from app.training.utils import pipeline_timed_context, split_x_y

if TYPE_CHECKING:
    from app.io.tracking import MLflowTracker

log = logging.getLogger(__name__)


def regression_main(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output: Path,
    tracker: "MLflowTracker | None" = None,
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
    tracker: MLflowTracker | None
        Optional tracker for experiment logging. Defaults to NullTracker (no-op).
    """
    from app.io.tracking import NullTracker, RegressionResult

    if tracker is None:
        tracker = NullTracker()

    train_x, train_y = split_x_y(train_df, "price", also_pop=["id"])
    test_x, test_y = split_x_y(test_df, "price")

    log.info("Regression fitting...")
    with pipeline_timed_context(pipeline) as pl:
        pl.fit(train_x, y=train_y)

    log.info("Regression predicting...")
    predicted_y: npt.NDArray = pipeline.predict(test_x)  # type: ignore[return]
    final_df = test_x.assign(price=predicted_y.tolist())
    final_df[["id", "price"]].to_csv(output, index=False)

    mse = metrics.mean_squared_error(test_y, predicted_y)
    mae = metrics.mean_absolute_error(test_y, predicted_y)
    r2 = metrics.r2_score(test_y, predicted_y)
    expl_var = metrics.explained_variance_score(test_y, predicted_y)
    max_err = metrics.max_error(test_y, predicted_y)
    med_err = metrics.median_absolute_error(test_y, predicted_y)

    if log.isEnabledFor(logging.INFO):
        for key, val in [
            ("MSE", mse), ("MAE", mae), ("R2", r2),
            ("Expl var", expl_var), ("Max err", max_err), ("Med err", med_err),
        ]:
            log.info("Regression %-10s: %.2f", key, val)

    # Extract hyperparams from the fitted pipeline
    lgbm = pipeline.named_steps["regression"].regressor_
    params = {
        "n_estimators": lgbm.n_estimators,
        "max_depth": lgbm.max_depth,
        "learning_rate": lgbm.learning_rate,
        "random_state": lgbm.random_state,
        "correlation_threshold": pipeline.named_steps["feature_processing"]
            .get_params()["numerical"]["feature_correlation"]["threshold"],
        "variance_threshold": pipeline.named_steps["feature_processing"]
            .get_params()["numerical"]["feature_variance"]["threshold"],
        "knn_neighbors": pipeline.named_steps["feature_processing"]
            .get_params()["numerical"]["imputer"]["n_neighbors"],
        "train_rows": int(train_x.shape[0]),
        "train_cols": int(train_x.shape[1]),
    }

    result = RegressionResult(
        r2=float(r2),
        mae=float(mae),
        mse=float(mse),
        rmse=math.sqrt(float(mse)),
        explained_variance=float(expl_var),
        max_error=float(max_err),
        median_absolute_error=float(med_err),
    )
    tracker.log_regression(params, result, pipeline)
