import numpy as np

from app.pipelines.base import SCHEMA_SPECIFIC_BASE_STEPS
from app.transformer import CustomRegressionFeatures, FeatureProcessingPipeline
from sklearn.pipeline import Pipeline
from app.transformer import CorrelationFeatureSelector, VarianceFeatureSelector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn import preprocessing
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor as LightGBMRegressor

# ----------
# REGRESSION
# ----------
PIPELINE_REGRESSION = Pipeline(
    steps=[
        *SCHEMA_SPECIFIC_BASE_STEPS,
        ("custom_features", CustomRegressionFeatures(sell_month_col="sell_month")),
        (
            "feature_processing",
            FeatureProcessingPipeline(
                numerical_steps=[
                    ("feature_correlation", CorrelationFeatureSelector(threshold=0.95)),
                    ("feature_variance", VarianceFeatureSelector(threshold=0.01)),
                    ("imputer", KNNImputer(n_neighbors=10, weights="distance")),
                    ("scaler", preprocessing.StandardScaler()),
                ],
                categorical_steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", preprocessing.OneHotEncoder(handle_unknown="ignore")),
                ],
            ),
        ),
        (
            "regression",
            TransformedTargetRegressor(
                regressor=LightGBMRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.08,
                    random_state=42,
                    verbosity=-1,
                ),
                transformer=preprocessing.FunctionTransformer(
                    np.log1p, inverse_func=np.expm1, validate=True, check_inverse=True
                ),
            ),
        ),
    ],
    memory=None,
)
