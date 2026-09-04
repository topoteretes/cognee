"""Blank multipart file parts must not 400 optional-upload routes.

Multipart has no encoding for an empty array, so clients that render a list
field with one blank item (Swagger UI) send ``data=""``. Before
``cognee.api.upload_fields`` the ``UploadFile`` validator rejected that part
before the handler ran, which made e.g. ``POST /v1/remember`` with
``content_type="code"`` unusable from Swagger UI.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cognee.api.upload_fields import drop_blank_uploads
from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.llm.routers.get_llm_router import get_llm_router
from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.users.methods import get_authenticated_user

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())

BLANK_PART = {"data": ("", b"", "application/octet-stream")}
PY_FILE = {"data": ("service.py", b"x = 1\n", "application/octet-stream")}


class FakeResult:
    status = "completed"

    def to_dict(self):
        return {"status": "completed"}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")
    app.include_router(get_add_router(), prefix="/add")
    app.include_router(get_llm_router(), prefix="/llm")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return TestClient(app)


@pytest.fixture
def fake_remember(monkeypatch):
    captured = {}

    async def _fake(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.remember"), "remember", _fake)
    return captured


@pytest.fixture
def fake_add(monkeypatch):
    captured = {}

    async def _fake(data, dataset_name=None, **kwargs):
        captured["data"] = data
        return SimpleNamespace(
            status="DATASET_PROCESSING_COMPLETED",
            pipeline_run_id=uuid4(),
            dataset_id=uuid4(),
            dataset_name=dataset_name,
            payload=None,
        )

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.add"), "add", _fake)
    return captured


# --- helper -----------------------------------------------------------------


def test_drop_blank_uploads_keeps_none_semantics():
    assert drop_blank_uploads(None) is None
    assert drop_blank_uploads([]) is None
    assert drop_blank_uploads(["", "  "]) is None


def test_drop_blank_uploads_keeps_real_uploads_in_order():
    first, second = object(), object()
    assert drop_blank_uploads([first, "", second]) == [first, second]


def test_drop_blank_uploads_rejects_text_with_actionable_message():
    with pytest.raises(HTTPException) as excinfo:
        drop_blank_uploads(["string"])
    assert excinfo.value.status_code == 400
    assert "'string'" in excinfo.value.detail
    assert "remove the item" in excinfo.value.detail


# --- /remember --------------------------------------------------------------


def test_remember_code_ingestion_with_blank_data_part(client, fake_remember):
    """The Swagger UI shape: content_type=code, raw_data filled, data left blank."""
    response = client.post(
        "/remember",
        files=BLANK_PART,
        data={
            "datasetName": "ds",
            "content_type": "code",
            "raw_data": "https://github.com/topoteretes/cognee",
            "node_set": "",
            "index_vectors": "false",
        },
    )
    assert response.status_code == 200, response.text
    assert fake_remember["data"] == ["https://github.com/topoteretes/cognee"]
    assert fake_remember["kwargs"]["content_type"] == "code"


def test_remember_blank_data_part_without_content_type_is_treated_as_no_uploads(
    client, fake_remember
):
    response = client.post("/remember", files=BLANK_PART, data={"datasetName": "ds"})
    assert response.status_code == 200, response.text
    assert fake_remember["data"] is None


def test_remember_placeholder_text_in_data_gets_clear_400(client, fake_remember):
    """Swagger UI's unfilled file item submits the literal text "string"."""
    response = client.post("/remember", data={"datasetName": "ds", "data": "string"})
    assert response.status_code == 400
    assert "accepts file uploads only" in response.json()["detail"]
    assert "'string'" in response.json()["detail"]


def test_remember_real_upload_still_reaches_handler_as_uploadfile(client, fake_remember):
    response = client.post("/remember", files=PY_FILE, data={"datasetName": "ds"})
    assert response.status_code == 200, response.text
    [upload] = fake_remember["data"]
    assert upload.filename == "service.py"
    assert hasattr(upload, "file")


def test_remember_code_still_rejects_real_upload(client, fake_remember):
    response = client.post(
        "/remember",
        files=PY_FILE,
        data={"datasetName": "ds", "content_type": "code", "raw_data": "/repo"},
    )
    assert response.status_code == 400
    assert "does not accept file uploads" in response.json()["detail"]


def test_remember_openapi_keeps_binary_schema_and_empty_list_examples(client):
    schema = client.app.openapi()
    body = schema["paths"]["/remember"]["post"]["requestBody"]["content"]["multipart/form-data"]
    body_schema = body["schema"]
    if "$ref" in body_schema:
        body_schema = schema["components"]["schemas"][body_schema["$ref"].rsplit("/", 1)[1]]
    props = body_schema["properties"]
    # Swagger UI must keep rendering a file picker for data.
    assert props["data"]["items"] == {"type": "string", "format": "binary"}
    # No field-level `null` example: Swagger UI would submit the text "null" as a tag.
    for name in ("node_set", "raw_data", "ontology_key"):
        assert props[name]["examples"] == [[]], name


# --- /add and /llm/infer-schema ---------------------------------------------


def test_add_blank_data_part_reaches_handler(client, fake_add):
    response = client.post("/add", files=BLANK_PART, data={"datasetName": "ds"})
    assert response.status_code == 200, response.text
    assert fake_add["data"] is None


def test_infer_schema_blank_data_part_is_no_files(client):
    response = client.post("/llm/infer-schema", files=BLANK_PART)
    assert response.status_code == 400
    assert response.json()["error"] == "Either text or at least one file must be provided."
