"""Azure Blob Storage integration tests.

These tests are skipped automatically when AZURE_STORAGE_CONNECTION_STRING is not set,
so they are safe to run in CI without Azure credentials.

Run manually after provisioning:
    cd backend
    AZURE_STORAGE_CONNECTION_STRING="..." uv run pytest tests/integration/test_azure_blob.py -v
"""

import io
import os
import uuid

import joblib
import pytest
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient


CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "models")

pytestmark = pytest.mark.skipif(
    not CONN_STR,
    reason="AZURE_STORAGE_CONNECTION_STRING not set — skipping Azure integration tests",
)


@pytest.fixture(scope="module")
def blob_service() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(CONN_STR)


@pytest.fixture(scope="module")
def container_client(blob_service: BlobServiceClient):
    client = blob_service.get_container_client(CONTAINER)
    try:
        client.create_container()
    except ResourceExistsError:
        pass  # already exists — that's fine
    return client


class TestAuthentication:
    def test_connection_string_is_valid(self, blob_service: BlobServiceClient) -> None:
        """Service client can be created from the connection string without raising."""
        props = blob_service.get_service_properties()
        assert props is not None

    def test_account_name_present(self, blob_service: BlobServiceClient) -> None:
        assert blob_service.account_name, "Account name should not be empty"


class TestContainerSetup:
    def test_container_exists(self, container_client) -> None:
        props = container_client.get_container_properties()
        assert props["name"] == CONTAINER

    def test_container_is_private(self, container_client) -> None:
        props = container_client.get_container_properties()
        # public_access should be None (private) for a models container
        assert props.get("public_access") is None, (
            f"Container '{CONTAINER}' has public access enabled — models should be private"
        )


class TestBlobRoundTrip:
    """Upload a small object, download it, verify it matches, then clean up."""

    def test_upload_and_download_bytes(self, container_client) -> None:
        blob_name = f"test-{uuid.uuid4().hex}.txt"
        payload = b"house-prediction-blob-test"
        try:
            container_client.upload_blob(blob_name, payload)
            downloaded = container_client.download_blob(blob_name).readall()
            assert downloaded == payload
        finally:
            container_client.delete_blob(blob_name, delete_snapshots="include")

    def test_joblib_serialization_round_trip(self, container_client) -> None:
        """Simulate saving and loading a model artifact the same way ModelArtifacts does."""
        blob_name = f"test-model-{uuid.uuid4().hex}.joblib"
        obj = {"model": "dummy", "params": [1, 2, 3]}

        buf = io.BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)

        try:
            container_client.upload_blob(blob_name, buf, overwrite=True)

            result_buf = io.BytesIO()
            container_client.download_blob(blob_name).readinto(result_buf)
            result_buf.seek(0)
            loaded = joblib.load(result_buf)

            assert loaded == obj
        finally:
            try:
                container_client.delete_blob(blob_name)
            except ResourceNotFoundError:
                pass

    def test_overwrite_existing_blob(self, container_client) -> None:
        blob_name = f"test-overwrite-{uuid.uuid4().hex}.txt"
        try:
            container_client.upload_blob(blob_name, b"v1")
            container_client.upload_blob(blob_name, b"v2", overwrite=True)
            content = container_client.download_blob(blob_name).readall()
            assert content == b"v2"
        finally:
            try:
                container_client.delete_blob(blob_name)
            except ResourceNotFoundError:
                pass


class TestModelArtifactsIntegration:
    """End-to-end test of the ModelArtifacts class itself against real Azure Blob."""

    def test_save_and_load_via_artifacts_class(self, tmp_path) -> None:
        from app.core.config import Settings
        from app.io.artifacts import ModelArtifacts

        settings = Settings(
            azure_storage_connection_string=CONN_STR,
            azure_storage_container=CONTAINER,
            model_dir=str(tmp_path),
        )
        artifacts = ModelArtifacts(settings)

        test_name = f"integration-test-{uuid.uuid4().hex}"
        payload = {"weights": [0.1, 0.2, 0.3], "bias": 1.5}

        try:
            artifacts.save(payload, test_name)
            loaded = artifacts.load(test_name)
            assert loaded == payload, f"Expected {payload}, got {loaded}"
        finally:
            # Clean up the blob we created
            try:
                service = BlobServiceClient.from_connection_string(CONN_STR)
                service.get_blob_client(
                    container=CONTAINER, blob=f"{test_name}.joblib"
                ).delete_blob()
            except Exception:
                pass
