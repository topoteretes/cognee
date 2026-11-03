"""remember() and cognify() send a custom graph_model to the remote instance (SDK-1387).

Before this, ``remember(data, graph_model=MyModel)`` under ``cognee.serve()`` returned
success while the server built the generic KnowledgeGraph: CloudClient.remember built
its form from a fixed field list that did not include ``graph_model``."""

import importlib
import json
from typing import Annotated, Literal

import pytest

from cognee.api.v1.serve import state as state_mod
from cognee.api.v1.serve.cloud_client import CloudClient, remote_graph_schema
from cognee.infrastructure.engine import DataPoint, Edge, FromIdentity
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import (
    fields_the_schema_cannot_carry,
    graph_schema_to_graph_model,
)

remember_mod = importlib.import_module("cognee.api.v1.remember.remember")
cognify_mod = importlib.import_module("cognee.api.v1.cognify.cognify")


class Language(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class LanguageGraph(DataPoint):
    languages: list[Language]


class InfluenceGraph(DataPoint):
    languages: list[Language]
    influences: list[Edge[Language, Language, Literal["influenced", "replaced"]]] = []


class Paradigm(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class TypedLanguage(DataPoint):
    name: str
    follows: Annotated[Paradigm, FromIdentity()]
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}


class ParadigmGraph(DataPoint):
    paradigms: list[Paradigm]
    languages: list[TypedLanguage]


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"status": "completed", "dataset_name": "main_dataset"}

    async def text(self):
        return ""


class _FakeSession:
    """Captures what the client POSTs: the form of a remember, the JSON of a cognify."""

    def __init__(self):
        self.calls = []

    def post(self, url, data=None, json=None, **kwargs):
        self.calls.append({"url": url, "form": data, "json": json})
        return _FakeResponse()

    async def close(self):
        pass


@pytest.fixture
def remote(monkeypatch):
    client = CloudClient("http://remote.invalid", "key")
    session = _FakeSession()

    async def get_session():
        return session

    monkeypatch.setattr(client, "_get_session", get_session)
    monkeypatch.setattr(state_mod, "_remote_client", client)
    return session


def _form_fields(form) -> dict:
    return {options["name"]: value for options, _headers, value in form._fields}


# ----- remember() -----


@pytest.mark.asyncio
async def test_remember_sends_the_model_as_the_schema_the_route_rebuilds(remote):
    await remember_mod.remember("Python was created by Guido.", graph_model=LanguageGraph)

    (call,) = remote.calls
    assert call["url"] == "http://remote.invalid/api/v1/remember"
    fields = _form_fields(call["form"])
    # POST /remember json.loads the field and hands it to graph_schema_to_graph_model:
    # what the client sends must be exactly what that pair accepts.
    rebuilt = graph_schema_to_graph_model(json.loads(fields["graph_model"]))
    assert rebuilt.__name__ == "LanguageGraph"
    assert "languages" in rebuilt.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_model", [None, KnowledgeGraph])
async def test_remember_sends_no_field_for_the_default_model(remote, graph_model):
    kwargs = {} if graph_model is None else {"graph_model": graph_model}

    await remember_mod.remember("Python was created by Guido.", **kwargs)

    (call,) = remote.calls
    assert "graph_model" not in _form_fields(call["form"])


@pytest.mark.asyncio
async def test_remember_refuses_a_model_with_typed_edges_before_any_upload(remote):
    with pytest.raises(ValueError, match=r"InfluenceGraph\.influences"):
        await remember_mod.remember("text", graph_model=InfluenceGraph)

    assert remote.calls == []


@pytest.mark.asyncio
async def test_remember_refuses_a_model_with_identity_references_before_any_upload(remote):
    with pytest.raises(ValueError, match=r"TypedLanguage\.follows"):
        await remember_mod.remember("text", graph_model=ParadigmGraph)

    assert remote.calls == []


# ----- cognify() -----


@pytest.mark.asyncio
async def test_cognify_sends_the_model_in_the_json_payload(remote):
    await cognify_mod.cognify(graph_model=LanguageGraph)

    (call,) = remote.calls
    assert call["url"] == "http://remote.invalid/api/v1/cognify"
    rebuilt = graph_schema_to_graph_model(call["json"]["graph_model"])
    assert rebuilt.__name__ == "LanguageGraph"


@pytest.mark.asyncio
async def test_cognify_sends_no_model_by_default(remote):
    await cognify_mod.cognify()

    (call,) = remote.calls
    assert "graph_model" not in call["json"]


@pytest.mark.asyncio
async def test_cognify_refuses_a_model_with_typed_edges(remote):
    with pytest.raises(ValueError, match=r"InfluenceGraph\.influences"):
        await cognify_mod.cognify(graph_model=InfluenceGraph)

    assert remote.calls == []


@pytest.mark.asyncio
async def test_cognify_refuses_an_ontology_file_path(remote):
    with pytest.raises(ValueError, match="ontology_file_path is not supported"):
        await cognify_mod.cognify(ontology_file_path="ontology.owl")

    assert remote.calls == []


# ----- the schema boundary -----


def test_schema_round_trip_keeps_a_plain_model():
    rebuilt = graph_schema_to_graph_model(remote_graph_schema(LanguageGraph))

    assert rebuilt.__name__ == "LanguageGraph"
    assert fields_the_schema_cannot_carry(LanguageGraph) == []


def test_the_detector_finds_edges_and_references_on_nested_types():
    assert fields_the_schema_cannot_carry(InfluenceGraph) == ["InfluenceGraph.influences"]
    assert fields_the_schema_cannot_carry(ParadigmGraph) == ["TypedLanguage.follows"]


def test_the_detector_names_why_the_schema_loses_them():
    """The regenerated model shows what the server would build: the edge rows as a
    nested DataPoint with string endpoints, the reference as a plain str."""
    from cognee.shared.graph_model_utils import graph_model_to_graph_schema

    rebuilt = graph_schema_to_graph_model(graph_model_to_graph_schema(InfluenceGraph))
    row = rebuilt.model_fields["influences"].annotation
    assert "Edge[" not in str(row)
    assert "InfluencesEdge" in str(row)
