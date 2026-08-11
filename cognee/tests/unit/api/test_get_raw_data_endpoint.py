import io
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.exceptions import DataNotFoundError
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture(scope="session")
def test_client():
    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router

    app = FastAPI()
    app.include_router(get_datasets_router(), prefix="/api/v1/datasets")

    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client):
    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    import importlib

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    datasets_router_module.send_telemetry = lambda *args, **kwargs: None

    test_client.app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _patch_raw_download_dependencies(
    monkeypatch, *, dataset_id, data_id, raw_data_location, name, mime_type
):
    """
    Patch the internal dataset/data lookups used by GET /datasets/{dataset_id}/data/{data_id}/raw.
    Keeps the test focused on response behavior (FileResponse vs StreamingResponse).
    """
    import importlib

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )

    monkeypatch.setattr(
        datasets_router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=dataset_id)]),
    )

    import cognee.modules.data.methods as data_methods_module

    monkeypatch.setattr(
        data_methods_module,
        "get_dataset_data",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=data_id,
                    owner_id=uuid.uuid4(),
                    raw_data_location=raw_data_location,
                    name=name,
                    mime_type=mime_type,
                )
            ]
        ),
    )


def test_get_raw_data_allows_authorized_dataset_reader(client, monkeypatch, tmp_path):
    """Dataset read access is sufficient even when the reader does not own the data row."""
    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()

    file_path = tmp_path / "example.txt"
    content = b"hello from disk"
    file_path.write_bytes(content)

    _patch_raw_download_dependencies(
        monkeypatch,
        dataset_id=dataset_id,
        data_id=data_id,
        raw_data_location=file_path.as_uri(),
        name="example.txt",
        mime_type="text/plain",
    )

    response = client.get(f"/api/v1/datasets/{dataset_id}/data/{data_id}/raw")
    assert response.status_code == 200
    assert response.content == content


def test_get_raw_data_s3_streams_bytes_without_s3_dependency(client, monkeypatch):
    """Streams bytes from an s3:// raw_data_location (mocked)."""
    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()

    _patch_raw_download_dependencies(
        monkeypatch,
        dataset_id=dataset_id,
        data_id=data_id,
        raw_data_location="s3://bucket/path/to/file.txt",
        name="file.txt",
        mime_type="text/plain",
    )

    import cognee.infrastructure.files.utils.open_data_file as open_data_file_module

    @asynccontextmanager
    async def fake_open_data_file(_file_path: str, mode: str = "rb", **_kwargs):
        assert mode == "rb"
        yield io.BytesIO(b"hello from s3")

    monkeypatch.setattr(open_data_file_module, "open_data_file", fake_open_data_file)

    response = client.get(f"/api/v1/datasets/{dataset_id}/data/{data_id}/raw")
    assert response.status_code == 200
    assert response.content == b"hello from s3"
    assert response.headers.get("content-disposition") == 'attachment; filename="file.txt"'


def test_get_raw_data_unsupported_scheme_returns_501(client, monkeypatch):
    """Returns 501 for unsupported raw_data_location schemes (e.g., http://)."""
    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()

    _patch_raw_download_dependencies(
        monkeypatch,
        dataset_id=dataset_id,
        data_id=data_id,
        raw_data_location="http://example.com/file.txt",
        name="file.txt",
        mime_type="text/plain",
    )

    response = client.get(f"/api/v1/datasets/{dataset_id}/data/{data_id}/raw")
    assert response.status_code == 501
    assert "Storage scheme 'http' not supported" in response.json()["detail"]


def test_get_raw_data_plain_path_downloads_bytes(client, monkeypatch, tmp_path):
    """Downloads bytes from a plain local path (no scheme)."""
    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()

    file_path = tmp_path / "plain.txt"
    content = b"plain content"
    file_path.write_bytes(content)

    _patch_raw_download_dependencies(
        monkeypatch,
        dataset_id=dataset_id,
        data_id=data_id,
        raw_data_location=str(file_path),
        name="plain.txt",
        mime_type="text/plain",
    )

    response = client.get(f"/api/v1/datasets/{dataset_id}/data/{data_id}/raw")
    assert response.status_code == 200
    assert response.content == content


def test_get_raw_data_encoded_path_downloads_bytes(client, monkeypatch, tmp_path):
    """Downloads bytes from a percent-encoded file URI (e.g. spaces)."""
    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()

    file_name = "example file.txt"
    file_path = tmp_path / file_name
    content = b"content with spaces"
    file_path.write_bytes(content)

    # Convert to URI, which should encode space as %20
    uri = file_path.as_uri()
    assert "%20" in uri

    _patch_raw_download_dependencies(
        monkeypatch,
        dataset_id=dataset_id,
        data_id=data_id,
        raw_data_location=uri,
        name=file_name,
        mime_type="text/plain",
    )

    response = client.get(f"/api/v1/datasets/{dataset_id}/data/{data_id}/raw")
    assert response.status_code == 200
    assert response.content == content


@pytest.mark.parametrize("permission_denied", [False, True])
@pytest.mark.parametrize("include_data_id", [False, True])
def test_dataset_read_endpoints_return_404_when_dataset_is_unavailable(
    client, monkeypatch, permission_denied, include_data_id
):
    import importlib

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    dataset_id = uuid.uuid4()
    data_id = uuid.uuid4()
    result = (
        AsyncMock(side_effect=PermissionDeniedError("denied"))
        if permission_denied
        else AsyncMock(return_value=[])
    )
    monkeypatch.setattr(datasets_router_module, "get_authorized_existing_datasets", result)

    path = f"/api/v1/datasets/{dataset_id}/data"
    if include_data_id:
        path += f"/{data_id}/raw"

    response = client.get(path)

    assert response.status_code == 404
    assert response.json() == {"message": f"Dataset ({dataset_id}) not found."}


def test_get_raw_data_rejects_data_from_another_dataset(client, monkeypatch):
    import importlib

    datasets_router_module = importlib.import_module(
        "cognee.api.v1.datasets.routers.get_datasets_router"
    )
    data_methods_module = importlib.import_module("cognee.modules.data.methods")
    dataset_id = uuid.uuid4()
    requested_data_id = uuid.uuid4()

    monkeypatch.setattr(
        datasets_router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=dataset_id)]),
    )
    monkeypatch.setattr(
        data_methods_module,
        "get_dataset_data",
        AsyncMock(return_value=[SimpleNamespace(id=uuid.uuid4())]),
    )

    with pytest.raises(DataNotFoundError):
        client.get(f"/api/v1/datasets/{dataset_id}/data/{requested_data_id}/raw")
