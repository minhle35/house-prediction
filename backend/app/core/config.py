from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Azure Blob Storage
    azure_storage_connection_string: str = ""
    azure_storage_container: str = "models"

    # Azure ML
    azure_ml_workspace: str = ""
    azure_ml_resource_group: str = ""
    azure_ml_subscription_id: str = ""

    # Model artifact names in Blob
    regression_model_name: str = "regression"
    classification_model_name: str = "classification"

    # App
    environment: str = "development"
    log_level: str = "INFO"

    # Local model cache / training fallback
    model_dir: str = "models"
    train_data_path: str = ""
    test_data_path: str = ""

    # MLflow experiment tracking
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = ""                                    # "" = local ./mlruns
    mlflow_experiment_regression: str = "house-price-regression"
    mlflow_experiment_classification: str = "house-type-classification"
    mlflow_register_models: bool = False                             # promote to Model Registry
    mlflow_r2_threshold: float = 0.70                                # min r2 to register regression
    mlflow_f1_threshold: float = 0.80                                # min weighted_f1 to register classification


settings = Settings()
