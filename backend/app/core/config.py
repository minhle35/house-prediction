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


settings = Settings()
