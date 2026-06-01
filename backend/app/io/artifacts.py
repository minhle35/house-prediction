import io
import logging
from pathlib import Path
from typing import Any

import joblib

from app.core.config import Settings

try:
    from azure.storage.blob import BlobServiceClient

    _AZURE_AVAILABLE = True
except ImportError:
    _AZURE_AVAILABLE = False

log = logging.getLogger(__name__)


class ModelArtifacts:
    """Save and load model artifacts — Azure Blob Storage with local joblib fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_dir = Path(settings.model_dir)

    def save(self, obj: Any, name: str) -> None:
        """Serialize obj with joblib → write to local model_dir → upload to Azure if configured."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        local = self._local_path(name)
        joblib.dump(obj, local)
        log.info("Saved '%s' to %s", name, local)

        if self._settings.azure_storage_connection_string:
            try:
                buf = io.BytesIO()
                joblib.dump(obj, buf)
                buf.seek(0)
                client = self._blob_client(name)
                client.upload_blob(buf, overwrite=True)
                log.info("Uploaded '%s' to Azure Blob", name)
            except Exception:
                log.warning("Failed to upload '%s' to Azure Blob", name, exc_info=True)

    def load(self, name: str) -> Any | None:
        """Try Azure Blob → local file → return None if both fail."""
        if self._settings.azure_storage_connection_string:
            obj = self._load_from_azure(name)
            if obj is not None:
                # cache locally so restarts skip Azure download
                self._model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(obj, self._local_path(name))
                return obj

        return self._load_from_local(name)

    def _load_from_azure(self, name: str) -> Any | None:
        if not _AZURE_AVAILABLE:
            log.warning("azure-storage-blob not installed; skipping Azure load for '%s'", name)
            return None
        try:
            client = self._blob_client(name)
            buf = io.BytesIO()
            client.download_blob().readinto(buf)
            buf.seek(0)
            obj = joblib.load(buf)
            log.info("Loaded '%s' from Azure Blob", name)
            return obj
        except Exception:
            log.warning("Could not load '%s' from Azure Blob", name, exc_info=True)
            return None

    def _load_from_local(self, name: str) -> Any | None:
        path = self._local_path(name)
        if not path.exists():
            return None
        try:
            obj = joblib.load(path)
            log.info("Loaded '%s' from local file %s", name, path)
            return obj
        except Exception:
            log.warning("Corrupt local file for '%s' at %s — deleting and treating as missing", name, path, exc_info=True)
            path.unlink(missing_ok=True)
            return None

    def _blob_client(self, name: str) -> Any:
        service = BlobServiceClient.from_connection_string(
            self._settings.azure_storage_connection_string
        )
        return service.get_blob_client(
            container=self._settings.azure_storage_container,
            blob=f"{name}.joblib",
        )

    def _local_path(self, name: str) -> Path:
        return self._model_dir / f"{name}.joblib"
