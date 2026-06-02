"""MLflow integration tests.

These tests run a training pipeline against a local MLflow file-store built
from a small slice of train.csv.  They are skipped automatically when
MLFLOW_TRACKING_URI is not set, so they are safe in CI without any MLflow server.

Run manually:
    cd backend
    MLFLOW_TRACKING_URI=sqlite:////tmp/test_mlruns.db \
      uv run pytest tests/integration/test_mlflow_tracking.py -v -m integration
"""

import os
from pathlib import Path

import pandas as pd
import pytest

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")
DATA_PATH = Path(__file__).parents[3] / "data" / "train.csv"

pytestmark = pytest.mark.skipif(
    not MLFLOW_URI,
    reason="MLFLOW_TRACKING_URI not set — skipping MLflow integration tests",
)


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """50-row slice of the real training data — enough to fit a pipeline fast."""
    return pd.read_csv(DATA_PATH).head(50)


@pytest.fixture(scope="module")
def tracker():
    from app.core.config import Settings
    from app.io.tracking import MLflowTracker

    s = Settings(
        mlflow_enabled=True,
        mlflow_tracking_uri=MLFLOW_URI,
        mlflow_register_models=False,
    )
    return MLflowTracker(s)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class TestRegressionTracking:
    def test_regression_run_recorded(self, tracker, small_df, tmp_path):
        import copy
        import mlflow
        from app.pipelines.regression import PIPELINE_REGRESSION
        from app.training.regression import regression_main

        pipeline = copy.deepcopy(PIPELINE_REGRESSION)
        regression_main(
            pipeline,
            small_df.copy(),
            small_df.copy(),
            tmp_path / "out.csv",
            tracker=tracker,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
        experiment = client.get_experiment_by_name("house-price-regression")
        assert experiment is not None

        runs = client.search_runs(experiment.experiment_id)
        assert len(runs) >= 1

    def test_regression_run_id_returned(self, tracker, small_df, tmp_path):
        import copy
        from app.pipelines.regression import PIPELINE_REGRESSION
        from app.training.regression import regression_main

        pipeline = copy.deepcopy(PIPELINE_REGRESSION)
        # Patch tracker to capture the run ID
        run_ids: list[str] = []
        original = tracker.log_regression

        def capturing_log(params, result, pipeline):
            rid = original(params, result, pipeline)
            if rid:
                run_ids.append(rid)
            return rid

        tracker.log_regression = capturing_log
        regression_main(
            pipeline,
            small_df.copy(),
            small_df.copy(),
            tmp_path / "out2.csv",
            tracker=tracker,
        )
        tracker.log_regression = original

        assert len(run_ids) == 1
        assert len(run_ids[0]) == 32

    def test_regression_model_loadable(self, tracker, small_df, tmp_path):
        import copy
        import mlflow
        import mlflow.sklearn
        from app.pipelines.regression import PIPELINE_REGRESSION
        from app.training.regression import regression_main

        pipeline = copy.deepcopy(PIPELINE_REGRESSION)
        run_ids: list[str] = []
        original = tracker.log_regression

        def capturing_log(params, result, p):
            rid = original(params, result, p)
            if rid:
                run_ids.append(rid)
            return rid

        tracker.log_regression = capturing_log
        regression_main(
            pipeline,
            small_df.copy(),
            small_df.copy(),
            tmp_path / "out3.csv",
            tracker=tracker,
        )
        tracker.log_regression = original

        assert run_ids, "No run ID captured"
        loaded = mlflow.sklearn.load_model(f"runs:/{run_ids[0]}/model")
        assert loaded is not None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassificationTracking:
    def test_classification_run_recorded(self, tracker, small_df, tmp_path):
        import copy
        import mlflow
        from app.pipelines.classification import PIPELINE_CLASSIFICATION
        from app.training.classification import classification_main

        pipeline = copy.deepcopy(PIPELINE_CLASSIFICATION)
        classification_main(
            pipeline,
            small_df.copy(),
            small_df.copy(),
            tmp_path / "cls_out.csv",
            tracker=tracker,
        )

        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
        experiment = client.get_experiment_by_name("house-type-classification")
        assert experiment is not None

        runs = client.search_runs(experiment.experiment_id)
        assert len(runs) >= 1

    def test_classification_metrics_present(self, tracker, small_df, tmp_path):
        import copy
        import mlflow
        from app.pipelines.classification import PIPELINE_CLASSIFICATION
        from app.training.classification import classification_main

        run_ids: list[str] = []
        original = tracker.log_classification

        def capturing_log(params, result, p, le):
            rid = original(params, result, p, le)
            if rid:
                run_ids.append(rid)
            return rid

        tracker.log_classification = capturing_log
        pipeline = copy.deepcopy(PIPELINE_CLASSIFICATION)
        classification_main(
            pipeline,
            small_df.copy(),
            small_df.copy(),
            tmp_path / "cls_out2.csv",
            tracker=tracker,
        )
        tracker.log_classification = original

        assert run_ids, "No run ID captured"
        run = mlflow.get_run(run_ids[0])
        assert "accuracy" in run.data.metrics
        assert "weighted_f1" in run.data.metrics
        assert "label_classes" in run.data.tags
