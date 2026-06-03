import argparse
import logging
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.core.logging import configure_logging
from app.io.artifacts import ModelArtifacts
from app.io.tracking import make_tracker
from app.pipelines import PIPELINE_CLASSIFICATION, PIPELINE_REGRESSION
from app.training import classification_main, regression_main

_artifacts = ModelArtifacts(settings)

log = logging.getLogger(__name__)


def run_regression(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    hline = "-" * 80
    log.info("\n%s\nStarting Regression Pipeline\n%s", hline, hline)
    tracker = make_tracker(settings)
    pipeline = regression_main(
        PIPELINE_REGRESSION, train_df.copy(), test_df.copy(), Path("regression.csv"),
        tracker=tracker,
    )
    _artifacts.save(pipeline, settings.regression_model_name)
    log.info("Regression model saved to %s", settings.regression_model_name)


def run_classification(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    hline = "-" * 80
    log.info("\n%s\nStarting Classification Pipeline\n%s", hline, hline)
    tracker = make_tracker(settings)
    pipeline, le = classification_main(
        PIPELINE_CLASSIFICATION,
        train_df.copy(),
        test_df.copy(),
        Path("classification.csv"),
        tracker=tracker,
    )
    _artifacts.save(pipeline, settings.classification_model_name)
    _artifacts.save(le, "label_encoder")
    log.info("Classification model saved to %s", settings.classification_model_name)


def main() -> None:
    cli = argparse.ArgumentParser(description="House Prediction — train and predict")
    cli.add_argument("train_data", type=Path, help="Path to training data CSV")
    cli.add_argument("test_data", type=Path, help="Path to test data CSV")
    cli.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Logging level (-v, -vv)"
    )
    cli.add_argument(
        "--model",
        choices=["regression", "classification", "both"],
        default="both",
        help="Which model(s) to run",
    )
    args = cli.parse_args()

    configure_logging(verbosity=args.verbosity)

    if not args.train_data.exists():
        raise FileNotFoundError(f"Training data not found: {args.train_data}")
    if not args.test_data.exists():
        raise FileNotFoundError(f"Test data not found: {args.test_data}")

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    if args.model in {"regression", "both"}:
        run_regression(train_df, test_df)
    if args.model in {"classification", "both"}:
        run_classification(train_df, test_df)


if __name__ == "__main__":
    main()
