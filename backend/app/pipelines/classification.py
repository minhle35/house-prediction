from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from app.pipelines.base import SCHEMA_SPECIFIC_BASE_STEPS
from app.transformer.feature_engineering import PropertyTypeFeatures
from app.transformer.selection import (
    FeatureProcessingPipeline,
    CorrelationFeatureSelector,
    VarianceFeatureSelector,
)
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# --------------
# CLASSIFICATION
# --------------
PIPELINE_CLASSIFICATION = Pipeline(
    steps=[
        *SCHEMA_SPECIFIC_BASE_STEPS,
        ("property_type_features", PropertyTypeFeatures()),
        (
            "feature_processing",
            FeatureProcessingPipeline(
                numerical_steps=[
                    ("feature_correlation", CorrelationFeatureSelector(threshold=0.95)),
                    ("feature_variance", VarianceFeatureSelector(threshold=0.01)),
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", preprocessing.StandardScaler()),
                ],
                categorical_steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", preprocessing.OneHotEncoder(handle_unknown="ignore")),
                ],
            ),
        ),
        ("classify", XGBClassifier()),
    ],
    memory=None,
)


def build_classification_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Build a classification pipeline with identical preprocessing to PIPELINE_CLASSIFICATION but a swappable estimator.

    Step name is always 'classify' so classify__sample_weight routes correctly
    via sklearn's fit_params passthrough, matching the production pipeline.
    """
    return Pipeline(
        steps=[
            *SCHEMA_SPECIFIC_BASE_STEPS,
            ("property_type_features", PropertyTypeFeatures()),
            (
                "feature_processing",
                FeatureProcessingPipeline(
                    numerical_steps=[
                        ("feature_correlation", CorrelationFeatureSelector(threshold=0.95)),
                        ("feature_variance", VarianceFeatureSelector(threshold=0.01)),
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", preprocessing.StandardScaler()),
                    ],
                    categorical_steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", preprocessing.OneHotEncoder(handle_unknown="ignore")),
                    ],
                ),
            ),
            ("classify", estimator),
        ],
        memory=None,
    )
