import logging
from timeit import default_timer
from typing import Iterator
import contextlib

import pandas as pd
import numpy.typing as npt
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


# ==================================================================================================
# Pipeline utilities
# ==================================================================================================
@contextlib.contextmanager
def pipeline_timed_context(pipeline: Pipeline) -> Iterator[Pipeline]:
    """Add timing information via logging for a pipeline.

    Notes
    -----
    We use contextlib.suppress to ensure the timing context does not cause any issues; we do not
    expect any actual failures here...
    """
    start = None
    with contextlib.suppress(Exception):
        if log.isEnabledFor(logging.INFO):
            start = default_timer()
            pipeline.verbose = True  # pyright: ignore[reportAttributeAccessIssue]

    yield pipeline

    with contextlib.suppress(Exception):
        if log.isEnabledFor(logging.INFO):
            if start is not None:
                log.info("Total fit time: %.2f seconds", default_timer() - start)
            pipeline.verbose = False  # pyright: ignore[reportAttributeAccessIssue]


# ==================================================================================================
# Util functions used to split dataframes into X,y for training and testing
# ==================================================================================================
def split_x_y(
    df: pd.DataFrame,
    y_col: str,
    also_pop: list[str] | None = None,
) -> tuple[pd.DataFrame, npt.NDArray]:
    """Split a dataframe into X,y

    Parameters
    ----------
    df: pandas.Dataframe
        The target dataframe
    y_col: str
        The y column to select from the df
    also_pop: list[str], optional
        List of columns to remove from the final X output.

    Returns
    -------
    tuple[pandas.Dataframe, numpy.typing.NDArray]
        Returns an X,y tuple.
    """
    y = df[y_col].to_numpy()
    x = df.drop(columns=[*(also_pop or []), y_col])
    return x, y
