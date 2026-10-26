"""Tests for dataset schema endpoints and related utilities."""

from uuid import uuid4

import pytest


def test_dataset_configuration_model_creation():
    """Verify DatasetConfiguration model can be instantiated with expected fields."""
    from cognee.modules.data.models.DatasetConfiguration import DatasetConfiguration

    config = DatasetConfiguration(
        dataset_id=uuid4(),
        graph_schema={"nodes": [{"name": "Person", "fields": ["name", "age"]}]},
        custom_prompt="Extract people and their relationships.",
    )

    assert config.dataset_id is not None
    assert config.graph_schema is not None
    assert config.custom_prompt == "Extract people and their relationships."


def test_dataset_configuration_model_defaults_to_none():
    """Verify optional fields default to None."""
    from cognee.modules.data.models.DatasetConfiguration import DatasetConfiguration

    config = DatasetConfiguration(dataset_id=uuid4())

    assert config.graph_schema is None
    assert config.custom_prompt is None


def test_prompt_templates_exist_and_render():
    """Verify that infer-schema and custom-prompt templates exist and render."""
    from cognee.infrastructure.llm.prompts import render_prompt

    custom_prompt_user = render_prompt(
        "custom_prompt_generation_user.txt",
        {"GRAPH_SCHEMA_JSON": '{"nodes": []}'},
    )
    assert len(custom_prompt_user) > 0
    assert "nodes" in custom_prompt_user

    custom_prompt_system = render_prompt("custom_prompt_generation_system.txt", {})
    assert len(custom_prompt_system) > 0

    infer_user = render_prompt("infer_schema_user.txt", {"SAMPLE_TEXT": "Hello world"})
    assert len(infer_user) > 0
    assert "Hello world" in infer_user

    infer_system = render_prompt("infer_schema_system.txt", {})
    assert len(infer_system) > 0


def test_safe_params_allowlist():
    """Verify _safe_params filters to allowed LLM parameters only."""
    from cognee.api.v1.llm.routers.get_llm_router import _safe_params

    params = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "seed": 42,
        "malicious_key": "drop table",
        "api_key": "secret",
    }

    result = _safe_params(params)

    assert result == {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "seed": 42,
    }
    assert "malicious_key" not in result
    assert "api_key" not in result


def test_safe_params_empty_input():
    """Verify _safe_params returns empty dict for empty input."""
    from cognee.api.v1.llm.routers.get_llm_router import _safe_params

    assert _safe_params({}) == {}


def test_dataset_schema_payload_dto_validation():
    """Verify DatasetSchemaPayloadDTO accepts valid input and defaults."""
    from cognee.api.v1.datasets.routers.get_datasets_router import DatasetSchemaPayloadDTO

    # Both fields provided
    dto = DatasetSchemaPayloadDTO(
        graph_schema={"nodes": []},
        custom_prompt="Extract entities.",
    )
    assert dto.graph_schema == {"nodes": []}
    assert dto.custom_prompt == "Extract entities."

    # Defaults to None
    dto_empty = DatasetSchemaPayloadDTO()
    assert dto_empty.graph_schema is None
    assert dto_empty.custom_prompt is None


def test_inferred_graph_schema_dto_accepts_json_schema_shape():
    """Verify InferredGraphSchemaDTO accepts required fields and $defs alias."""
    from cognee.api.v1.llm.routers.get_llm_router import InferredGraphSchemaDTO

    dto = InferredGraphSchemaDTO.model_validate(
        {
            "title": "CompanyGraph",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "$defs": {
                "Person": {
                    "title": "Person",
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "additional_json_schema_keyword": True,
        }
    )

    dumped = dto.model_dump(by_alias=True)
    assert dumped["$defs"]["Person"]["title"] == "Person"
    assert dumped["additional_json_schema_keyword"] is True


def test_inferred_graph_schema_dto_requires_core_fields():
    """Verify InferredGraphSchemaDTO enforces title/type/properties fields."""
    from pydantic import ValidationError

    from cognee.api.v1.llm.routers.get_llm_router import InferredGraphSchemaDTO

    with pytest.raises(ValidationError):
        InferredGraphSchemaDTO.model_validate({"type": "object", "properties": {}})


def test_sample_text_short_input_unchanged():
    """Short text should pass through unchanged."""
    from cognee.api.v1.llm.routers.get_llm_router import _sample_text

    short = "Hello world, this is a short text."
    assert _sample_text(short) == short


def test_sample_text_slightly_over_limit_uses_two_sections():
    """Text between 1x and 2x the limit should use begin+end (no middle)."""
    from cognee.api.v1.llm.routers.get_llm_router import _sample_text

    # 150 chars with a 100-char limit → two-section strategy
    text = "A" * 75 + "B" * 75
    result = _sample_text(text, max_chars=100)

    assert result.count("[...]") == 1  # only one separator
    assert result.startswith("A")
    assert result.endswith("B")
    assert len(result) <= 100


def test_sample_text_large_input_uses_three_sections():
    """Text over 2x the limit should use begin/middle/end."""
    from cognee.api.v1.llm.routers.get_llm_router import _sample_text

    long_text = "A" * 5_000 + "B" * 5_000 + "C" * 5_000
    result = _sample_text(long_text, max_chars=3_000)

    assert result.count("[...]") == 2  # two separators
    assert result.startswith("A")
    assert result.endswith("C")
    assert "B" in result
    assert len(result) <= 3_000


def test_sample_text_exact_boundary():
    """Text exactly at the limit should not be sampled."""
    from cognee.api.v1.llm.routers.get_llm_router import _sample_text

    text = "X" * 100
    assert _sample_text(text, max_chars=100) == text


# ── PATCH /datasets/{id}/schema ──────────────────────────────────────────────


STORED_SCHEMA = {
    "options": {"autoTypeNodes": False},
    "entities": [
        {
            "name": "Person",
            "description": "A person.",
            "fields": [
                {"kind": "primitive", "name": "age", "primitiveType": "number"},
                {
                    "kind": "relation",
                    "name": "works_at",
                    "relation": {"targetEntityName": "Organization", "cardinality": "one"},
                },
            ],
        },
        {"name": "Organization", "fields": []},
    ],
}


class _SchemaStore:
    """In-memory stand-in for the dataset_configurations table.

    The router reads and writes through two module-level helpers shared by GET,
    PUT and PATCH; faking those keeps the handlers under test while the SQL
    stays identical to what PUT already ran.
    """

    def __init__(self):
        self.rows: dict = {}
        self.writes: list = []

    async def read(self, dataset_id):
        row = self.rows.get(dataset_id, {"graph_schema": None, "custom_prompt": None})
        return dict(row)

    async def store(self, dataset_id, graph_schema, custom_prompt):
        self.writes.append((dataset_id, graph_schema, custom_prompt))
        row = self.rows.setdefault(dataset_id, {"graph_schema": None, "custom_prompt": None})
        if graph_schema is not None:
            row["graph_schema"] = graph_schema
        if custom_prompt is not None:
            row["custom_prompt"] = custom_prompt


@pytest.fixture
def schema_api(monkeypatch):
    """TestClient over the datasets router with auth and storage faked.

    Returns ``(client, dataset_id, store)``. ``store.authorized`` controls what
    the permission check answers: a list of datasets, ``[]`` (not found) or an
    exception instance to raise.
    """
    from importlib import import_module
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from cognee.exceptions import CogneeApiError

    router_module = import_module("cognee.api.v1.datasets.routers.get_datasets_router")
    dataset_id = uuid4()
    store = _SchemaStore()
    store.authorized = [SimpleNamespace(id=dataset_id)]
    store.permissions_asked = []

    async def fake_authorized_datasets(dataset_ids, permission, _user):
        store.permissions_asked.append(permission)
        if isinstance(store.authorized, Exception):
            raise store.authorized
        return store.authorized

    monkeypatch.setattr(router_module, "get_authorized_existing_datasets", fake_authorized_datasets)
    monkeypatch.setattr(router_module, "_read_dataset_schema", store.read)
    monkeypatch.setattr(router_module, "_store_dataset_schema", store.store)

    app = FastAPI()
    app.include_router(router_module.get_datasets_router(), prefix="/api/v1/datasets")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )

    # Same mapping cognee/api/client.py applies: a CogneeApiError's own status code.
    @app.exception_handler(CogneeApiError)
    async def _cognee_error(_request, exc: CogneeApiError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    return TestClient(app), dataset_id, store


def _entity(document, name):
    return next(item for item in document["entities"] if item["name"] == name)


def _field_names(document, name):
    return [item["name"] for item in _entity(document, name)["fields"]]


def test_patch_schema_adds_entity(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": "Extract."}

    response = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={"entities": [{"name": "City", "fields": [{"kind": "primitive", "name": "zip"}]}]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["name"] for item in body["graph_schema"]["entities"]] == [
        "Person",
        "Organization",
        "City",
    ]
    assert body["custom_prompt"] == "Extract."
    assert store.permissions_asked == ["write"]


def test_patch_schema_adds_field_to_existing_entity(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": None}

    response = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={
            "entities": [
                {
                    "name": "Person",
                    "fields": [{"kind": "enum", "name": "mood", "enumValues": ["happy", "sad"]}],
                }
            ]
        },
    )

    assert response.status_code == 200, response.text
    merged = response.json()["graph_schema"]
    assert _field_names(merged, "Person") == ["age", "works_at", "mood"]
    assert _entity(merged, "Person")["description"] == "A person."


def test_patch_schema_replaces_field_by_name(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": None}

    response = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={
            "entities": [
                {
                    "name": "Person",
                    "fields": [{"kind": "primitive", "name": "age", "primitive_type": "string"}],
                }
            ]
        },
    )

    assert response.status_code == 200, response.text
    merged = response.json()["graph_schema"]
    assert _field_names(merged, "Person") == ["age", "works_at"]
    assert _entity(merged, "Person")["fields"][0]["primitiveType"] == "string"


def test_patch_schema_removes_field(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": None}

    response = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={"removeFields": [{"entity": "Person", "field": "age"}]},
    )

    assert response.status_code == 200, response.text
    assert _field_names(response.json()["graph_schema"], "Person") == ["works_at"]


def test_patch_schema_removes_entity(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": None}

    response = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={
            "remove_fields": [{"entity": "Person", "field": "works_at"}],
            "remove_entities": ["Organization"],
        },
    )

    assert response.status_code == 200, response.text
    assert [item["name"] for item in response.json()["graph_schema"]["entities"]] == ["Person"]


def test_patch_schema_rejects_removing_relation_target(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": None}

    response = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={"removeEntities": ["Organization"]},
    )

    assert response.status_code == 422, response.text
    assert "'Organization'" in response.json()["error"]
    assert "Person.works_at" in response.json()["error"]
    assert store.writes == []
    assert store.rows[dataset_id]["graph_schema"] == STORED_SCHEMA


def test_patch_schema_without_stored_schema_equals_put(schema_api):
    client, dataset_id, store = schema_api
    body = {
        "options": {"autoTypeNodes": False},
        "entities": [{"name": "Person", "fields": [{"kind": "primitive", "name": "age"}]}],
    }

    patched = client.patch(f"/api/v1/datasets/{dataset_id}/schema", json=body)
    assert patched.status_code == 200, patched.text
    via_patch = store.rows[dataset_id]["graph_schema"]

    from cognee.modules.graph_models import GraphSchemaSpec

    canonical = GraphSchemaSpec.model_validate(body).model_dump(by_alias=True, exclude_none=True)
    put = client.put(f"/api/v1/datasets/{dataset_id}/schema", json={"graphSchema": canonical})
    assert put.status_code == 200, put.text

    assert via_patch == store.rows[dataset_id]["graph_schema"] == canonical
    assert patched.json()["graph_schema"] == canonical


def test_patch_schema_rejects_invalid_merged_spec(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": None}

    response = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={"entities": [{"name": "bad-name"}]},
    )

    assert response.status_code == 422, response.text
    assert "'bad-name'" in response.json()["error"]
    assert "plain identifier" in response.json()["error"]
    assert store.writes == []


def test_patch_schema_then_get_returns_merged_document(schema_api):
    client, dataset_id, store = schema_api
    store.rows[dataset_id] = {"graph_schema": STORED_SCHEMA, "custom_prompt": "Extract."}

    patched = client.patch(
        f"/api/v1/datasets/{dataset_id}/schema",
        json={"entities": [{"name": "City"}], "options": {"autoTypeNodes": True}},
    )
    assert patched.status_code == 200, patched.text

    fetched = client.get(f"/api/v1/datasets/{dataset_id}/schema")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json() == patched.json()
    assert fetched.json()["graph_schema"]["options"] == {"autoTypeNodes": True}
    assert fetched.json()["custom_prompt"] == "Extract."
    assert [item["name"] for item in fetched.json()["graph_schema"]["entities"]] == [
        "Person",
        "Organization",
        "City",
    ]


def test_patch_schema_unauthorized_matches_put(schema_api):
    from cognee.modules.users.exceptions import PermissionDeniedError

    client, dataset_id, store = schema_api
    patch_body = {"entities": [{"name": "City"}]}
    put_body = {"graphSchema": STORED_SCHEMA}

    store.authorized = PermissionDeniedError("no write permission", log=False)
    denied_patch = client.patch(f"/api/v1/datasets/{dataset_id}/schema", json=patch_body)
    denied_put = client.put(f"/api/v1/datasets/{dataset_id}/schema", json=put_body)
    assert denied_patch.status_code == denied_put.status_code == 403

    store.authorized = []
    missing_patch = client.patch(f"/api/v1/datasets/{dataset_id}/schema", json=patch_body)
    missing_put = client.put(f"/api/v1/datasets/{dataset_id}/schema", json=put_body)
    assert missing_patch.status_code == missing_put.status_code == 404

    assert store.writes == []
    assert set(store.permissions_asked) == {"write"}
