"""Journey: a custom graph model sent over HTTP as its JSON schema builds that graph.

``POST /api/v1/cognify`` and ``POST /api/v1/remember`` forward the schema to the SDK,
which builds the model class itself. Through the real app and real local databases:
the extraction is asked for the schema's model, the dataset graph holds its node types,
and a schema that cannot be built is a 400 that stores nothing, blocking or background.

In mock mode the LLM answer for the schema's model is scripted; everything else is the
journeys' deterministic mock.
"""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import BaseModel

from cognee.shared.graph_model_utils import graph_model_to_graph_schema
from cognee.tests.journeys import _support
from cognee.tests.journeys.test_http_api_journey import Session, api

pytestmark = pytest.mark.journey


class Book(BaseModel):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class Library(BaseModel):
    name: str
    books: list[Book] = []
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


SCHEMA = graph_model_to_graph_schema(Library)

COGNIFY_DOC = (
    "Title: Harrowgate Library\n\nThe Harrowgate Library lends out The Tide Tables "
    "and The Lighthouse Keeper."
)
REMEMBER_DOC = "Title: Pellworth Library\n\nThe Pellworth Library holds Cheese of the Ridge."

# What the extraction answers for the Library model, per document.
ANSWERS = {
    "Harrowgate Library": {
        "name": "Harrowgate Library",
        "books": [{"name": "The Tide Tables"}, {"name": "The Lighthouse Keeper"}],
    },
    "Pellworth Library": {"name": "Pellworth Library", "books": [{"name": "Cheese of the Ridge"}]},
}

EXTERNAL_REF = {
    "title": "Refused",
    "type": "object",
    "properties": {"nodes": {"type": "array", "items": {"$ref": "/etc/passwd"}}},
}


@pytest.fixture
def library_llm(monkeypatch, mock_llm):
    """Answer the extraction for the schema's model; defer everything else to the mock."""
    if mock_llm is None:
        yield []
        return

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.tests.journeys.mock_ai import title_of

    deferred = LLMGateway.acreate_structured_output
    asked: list[str] = []

    async def acreate(text_input, system_prompt, response_model, **kwargs):
        if getattr(response_model, "__name__", None) == "Library":
            title = title_of(text_input)
            asked.append(title)
            return response_model.model_validate(ANSWERS[title])
        return await deferred(text_input, system_prompt, response_model, **kwargs)

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", staticmethod(acreate))
    yield asked


def _graph(user: Session, dataset_name: str) -> dict[str, set[str]]:
    dataset = user.dataset(dataset_name)
    assert dataset is not None, f"{dataset_name} not listed"
    response = user.get(f"/api/v1/datasets/{dataset['id']}/graph")
    assert response.status_code == 200, response.text
    found: dict[str, set[str]] = {}
    for node in response.json()["nodes"]:
        found.setdefault(node["type"], set()).add(node["label"])
    return found


def _assert_library_graph(found: dict[str, set[str]], library: str, books: set[str]) -> None:
    assert library in found.get("Library", set()), f"node types: {found}"
    if _support.IS_MOCK:
        assert books <= found.get("Book", set()), f"node types: {found}"
    else:
        assert found.get("Book"), f"node types: {found}"


def test_a_schema_sent_over_http_builds_its_graph(api, library_llm):  # noqa: F811
    unique = uuid.uuid4().hex[:8]
    user = Session(api, f"schema-{unique}@cognee-journeys.org").register_and_login()

    # --- POST /cognify with the schema in the JSON body -------------------------------
    added = user.post(
        "/api/v1/add", data={"raw_data": [COGNIFY_DOC], "datasetName": "schema_cognify"}
    )
    assert added.status_code in (200, 201), added.text
    cognified = user.post(
        "/api/v1/cognify",
        json={"datasets": ["schema_cognify"], "graphModel": SCHEMA, "runInBackground": False},
    )
    assert cognified.status_code == 200, cognified.text
    _assert_library_graph(
        _graph(user, "schema_cognify"),
        "Harrowgate Library",
        {"The Tide Tables", "The Lighthouse Keeper"},
    )

    # --- POST /remember with the schema as the form field's JSON text -----------------
    remembered = user.post(
        "/api/v1/remember",
        data={
            "raw_data": [REMEMBER_DOC],
            "datasetName": "schema_remember",
            "graph_model": json.dumps(SCHEMA),
            "self_improvement": "false",
        },
    )
    assert remembered.status_code == 200, remembered.text
    _assert_library_graph(
        _graph(user, "schema_remember"), "Pellworth Library", {"Cheese of the Ridge"}
    )

    if _support.IS_MOCK:
        # The extraction was asked for the schema's own model, once per document.
        assert sorted(library_llm) == ["Harrowgate Library", "Pellworth Library"]


@pytest.mark.parametrize("run_in_background", [False, True])
def test_an_unbuildable_schema_is_a_400_that_stores_nothing(api, run_in_background):  # noqa: F811
    unique = uuid.uuid4().hex[:8]
    user = Session(api, f"refused-{unique}@cognee-journeys.org").register_and_login()

    refused = user.post(
        "/api/v1/remember",
        data={
            "raw_data": [REMEMBER_DOC],
            "datasetName": "schema_refused",
            "graph_model": json.dumps(EXTERNAL_REF),
            "run_in_background": str(run_in_background).lower(),
        },
    )
    assert refused.status_code == 400, refused.text
    assert "ExternalSchemaReferenceError" in refused.json()["detail"]
    assert user.dataset("schema_refused") is None

    # The cognify route refuses it before a pipeline run is created for the dataset.
    added = user.post("/api/v1/add", data={"raw_data": [COGNIFY_DOC], "datasetName": "staged"})
    assert added.status_code in (200, 201), added.text
    dataset_id = user.dataset("staged")["id"]
    refused = user.post(
        "/api/v1/cognify",
        json={
            "datasets": ["staged"],
            "graphModel": {key: value for key, value in SCHEMA.items() if key != "title"},
            "runInBackground": run_in_background,
        },
    )
    assert refused.status_code == 400, refused.text
    assert "InvalidGraphSchemaError" in refused.json()["detail"]
    assert user.status(dataset_id) is None
