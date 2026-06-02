import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn import metrics, preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import numpy.typing as npt

from app.training.utils import split_x_y, pipeline_timed_context

if TYPE_CHECKING:
    from app.io.tracking import MLflowTracker

log = logging.getLogger(__name__)


def classification_main(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output: Path,
    tracker: "MLflowTracker | None" = None,
) -> None:
    """Fit a classification model over training data and predict for test data.

    Parameters
    ----------
    train_df: pandas.Dataframe
        The dataframe to train over. This will be split with `split_x_y` with `y_col="type"`.
    test_df: pandas.Dataframe
        The dataframe to predict for. The y values are ignored here. Must have an ID column for the
        output.
    output: Path
        The output file to save the predictions over `test_df` to
    tracker: MLflowTracker | None
        Optional tracker for experiment logging. Defaults to NullTracker (no-op).
    """
    from app.io.tracking import ClassificationResult, NullTracker

    if tracker is None:
        tracker = NullTracker()

    train_x, train_y = split_x_y(train_df, "type", also_pop=["id"])
    test_x, test_y = split_x_y(test_df, "type")

    le = preprocessing.LabelEncoder()
    train_y_encoded: npt.NDArray[int] = le.fit_transform(train_y)  # type:ignore[return]
    test_y_encoded: npt.NDArray[int] = le.transform(test_y)  # type:ignore[return]

    class_counts = pd.Series(train_y).value_counts()
    log.info("Original class distribution:\n%s", class_counts)

    majority_class_count = class_counts.max()
    class_weights: dict[int, float] = {}

    for class_name, count in class_counts.items():
        class_idx = int(le.transform([class_name])[0])
        class_ratio = count / majority_class_count
        log.info("Class '%s' count, ratio: %d - %d", class_name, count, class_ratio)

        if class_ratio < 0.01:
            weight = 10.0
            log.info("Class ratio '%s' < 0.01 is very rare, applying high weight: %d", class_name, weight)
        elif class_ratio < 0.05:
            weight = 5.0
            log.info("Class ratio '%s' < 0.05 is rare, applying high weight: %d", class_name, weight)
        elif class_ratio < 0.1:
            weight = 3.0
            log.info("Class ratio '%s' < 0.1 is uncommon, applying moderate weight: %d", class_name, weight)
        else:
            weight = 1.0

        class_weights[class_idx] = weight

    log.info("Class weights: %s", class_weights)

    sample_weights = np.array([class_weights.get(cls, 1.0) for cls in train_y_encoded])

    with pipeline_timed_context(pipeline) as pl:
        pl.fit(train_x, y=train_y_encoded, classify__sample_weight=sample_weights)

    log.info("Classification predicting...")
    predicted_y: npt.NDArray[int] = pipeline.predict(test_x)  # type:ignore[return]
    predicted_y_decoded = le.inverse_transform(predicted_y.tolist())
    final_df = test_x.assign(type=le.inverse_transform(predicted_y.tolist()))
    final_df[["id", "type"]].to_csv(output, index=False)

    accuracy = float(metrics.accuracy_score(test_y_encoded, predicted_y))
    weighted_f1 = float(metrics.f1_score(test_y_encoded, predicted_y, average="weighted", zero_division=1))
    macro_f1 = float(metrics.f1_score(test_y_encoded, predicted_y, average="macro", zero_division=1))
    precision = float(metrics.precision_score(test_y_encoded, predicted_y, average="weighted", zero_division=1))
    recall = float(metrics.recall_score(test_y_encoded, predicted_y, average="weighted", zero_division=1))
    cohen_kappa = float(metrics.cohen_kappa_score(test_y_encoded, predicted_y))
    mcc = float(metrics.matthews_corrcoef(test_y_encoded, predicted_y))

    f1_per_class_arr = metrics.f1_score(test_y_encoded, predicted_y, average=None, zero_division=1)
    per_class_f1 = {str(cls): float(score) for cls, score in zip(le.classes_, f1_per_class_arr, strict=False)}

    if log.isEnabledFor(logging.INFO):
        for key, val in [
            ("Accuracy", accuracy), ("Weighted F1", weighted_f1), ("Macro F1", macro_f1),
            ("Precision", precision), ("Recall", recall),
            ("Cohen Kappa", cohen_kappa), ("Matthews Corr. Coef.", mcc),
        ]:
            log.info("Classification %-20s: %.4f", key, val)

        cm = metrics.confusion_matrix(test_y_encoded, predicted_y)
        log.info("Classification Confusion Matrix:\nClasses: %s\n%s", le.classes_, cm)

        cr = metrics.classification_report(test_y, predicted_y_decoded, zero_division=1)
        log.info("Classification Report:\n%s", cr)

        log.info("F1 Scores per class:")
        for cls, score in per_class_f1.items():
            log.info("F1 Score for class '%s': %.4f", cls, score)

    params = {
        "correlation_threshold": pipeline.named_steps["feature_processing"]
            .get_params()["numerical"]["feature_correlation"]["threshold"],
        "variance_threshold": pipeline.named_steps["feature_processing"]
            .get_params()["numerical"]["feature_variance"]["threshold"],
        "train_rows": int(train_x.shape[0]),
        "num_classes": len(le.classes_),
        "class_weights": json.dumps({str(k): v for k, v in class_weights.items()}),
    }

    result = ClassificationResult(
        accuracy=accuracy,
        weighted_f1=weighted_f1,
        macro_f1=macro_f1,
        precision=precision,
        recall=recall,
        cohen_kappa=cohen_kappa,
        matthews_corrcoef=mcc,
        per_class_f1=per_class_f1,
    )
    tracker.log_classification(params, result, pipeline, le)
