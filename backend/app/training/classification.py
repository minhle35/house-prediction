import functools
import logging
from pathlib import Path
from sklearn import metrics, preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import numpy.typing as npt

from app.training.utils import split_x_y, pipeline_timed_context

log = logging.getLogger(__name__)


def classification_main(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output: Path,
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
    """

    train_x, train_y = split_x_y(train_df, "type", also_pop=["id"])
    test_x, test_y = split_x_y(test_df, "type")

    le = preprocessing.LabelEncoder()
    train_y_encoded: npt.NDArray[int] = le.fit_transform(train_y)  # type:ignore[return]
    test_y_encoded: npt.NDArray[int] = le.transform(test_y)  # type:ignore[return]

    # Get class counts to determine weights
    class_counts = pd.Series(train_y).value_counts()
    log.info("Original class distribution:\n%s", class_counts)

    # Calculate custom weights inversely proportional to class frequency
    # with extra boost for very rare classes
    majority_class_count = class_counts.max()
    class_weights = {}

    for class_name, count in class_counts.items():
        class_idx = le.transform([class_name])[0]
        class_ratio = count / majority_class_count
        log.info("Class '%s' count, ratio: %d - %d", class_name, count, class_ratio)

        # Apply stronger weights for rarer classes
        if class_ratio < 0.01:  # Very rare classes (<1% of majority)
            weight = 10.0  # Very high weight
            log.info(
                "Class ratio '%s' < 0.01 is very rare, applying high weight: %d",
                class_name,
                weight,
            )
        elif class_ratio < 0.05:  # Rare classes (<5% of majority)
            weight = 5.0  # High weight
            log.info(
                "Class ratio '%s' < 0.05 is rare, applying high weight: %d",
                class_name,
                weight,
            )
        elif class_ratio < 0.1:  # Uncommon classes (<10% of majority)
            weight = 3.0  # Moderate weight
            log.info(
                "Class ratio '%s' < 0.1 is uncommon, applying moderate weight: %d",
                class_name,
                weight,
            )
        else:  # Common classes
            weight = 1.0  # Regular weight

        class_weights[class_idx] = weight

    log.info("Class weights: %s", class_weights)

    # Generate sample weights for XGBoost to use during training
    sample_weights = np.array([class_weights.get(cls, 1.0) for cls in train_y_encoded])

    with pipeline_timed_context(pipeline) as pl:  # For debugging; use `-v`...
        pl.fit(train_x, y=train_y_encoded, classify__sample_weight=sample_weights)

    log.info("Classification predicting...")
    predicted_y: npt.NDArray[int] = pipeline.predict(test_x)  # type:ignore[return]
    predicted_y_decoded = le.inverse_transform(predicted_y.tolist())
    final_df = test_x.assign(type=le.inverse_transform(predicted_y.tolist()))
    final_df[["id", "type"]].to_csv(output, index=False)

    if log.isEnabledFor(logging.INFO):  # For debugging; use `-v`...
        mm = {
            "Accuracy": metrics.accuracy_score,
            "Weighted F1": functools.partial(
                metrics.f1_score, average="weighted", zero_division=1
            ),
            "Macro F1": functools.partial(
                metrics.f1_score, average="macro", zero_division=1
            ),
            "Precision": functools.partial(
                metrics.precision_score, average="weighted", zero_division=1
            ),
            "Recall": functools.partial(
                metrics.recall_score, average="weighted", zero_division=1
            ),
            "Cohen Kappa": metrics.cohen_kappa_score,
            "Matthews Corr. Coef.": metrics.matthews_corrcoef,
        }
        for key, fn in mm.items():
            val = f"{fn(test_y_encoded, predicted_y):,.4f}"
            log.info("Classification %-20s: %s", key, val)

        # Create confusion matrix with original class names
        cm = metrics.confusion_matrix(test_y_encoded, predicted_y)
        class_names = le.classes_
        log.info("Classification Confusion Matrix (original class names):")
        log.info("Classes: %s", class_names)
        log.info("\n%s", cm)

        # Create classification report with original class names
        cr = metrics.classification_report(test_y, predicted_y_decoded, zero_division=1)
        log.info("Classification Report (original class names):\n%s", cr)

        # Create a DataFrame for better formatted report
        cr_dict = metrics.classification_report(
            test_y, predicted_y_decoded, output_dict=True, zero_division=1
        )
        cr_df = pd.DataFrame(cr_dict).T
        cr_df = cr_df.round(4)
        log.info("Classification Report as DataFrame:\n%s", cr_df)

        # Print F1 scores for each class with original class names
        f1_per_class = metrics.f1_score(
            test_y_encoded, predicted_y, average=None, zero_division=1
        )
        f1_per_class_dict = {}

        log.info("F1 Scores per class:")
        for cls, score in zip(le.classes_, f1_per_class, strict=False):
            log.info(f"F1 Score for class '{cls}': {score:.4f}")
            f1_per_class_dict[cls] = score
