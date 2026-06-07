"""CLI: python -m experiments.run_classification --train-data [file path] --test-data [file path]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_BACKEND_DIR = Path(__file__).parents[1]
_DEFAULT_URI = f"sqlite:///{_BACKEND_DIR}/mlflow.db"
_DEFAULT_EXPERIMENT = "house-type-classification-search"
_DEFAULT_CLASSIFICATION_CONFIG = (
    Path(__file__).parent / "candidates" / "classification.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep classification model candidates and log to MLflow."
    )
    p.add_argument("--train-data", required=True, type=Path, metavar="PATH")
    p.add_argument("--test-data", required=True, type=Path, metavar="PATH")
    p.add_argument("--experiment-name", default=_DEFAULT_EXPERIMENT, metavar="STR")
    p.add_argument("--tracking-uri", default=_DEFAULT_URI, metavar="URI")
    p.add_argument(
        "--config",
        default=_DEFAULT_CLASSIFICATION_CONFIG,
        type=Path,
        metavar="PATH",
        help="YAML candidates config (default: candidates/classification.yaml)",
    )
    p.add_argument(
        "--log-artifacts",
        action="store_true",
        default=False,
        help="Save fitted pipelines as MLflow model artifacts (slow, off by default)",
    )
    p.add_argument("--random-state", type=int, default=42, metavar="INT")
    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v INFO, -vv DEBUG"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-8s %(name)s %(message)s"
    )

    from experiments.candidates.loader import load_classification_candidates
    from experiments.runner import ClassificationRunner

    candidates = load_classification_candidates(config_path=args.config)
    logging.getLogger(__name__).info(
        "Loaded %d families from %s", len(candidates), args.config
    )

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    runner = ClassificationRunner(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        log_artifacts=args.log_artifacts,
        random_state=args.random_state,
        config_path=args.config,
    )
    runner.run(train_df, test_df, candidates)


if __name__ == "__main__":
    main(sys.argv[1:])
