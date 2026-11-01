"""A graph_model schema must not name a type that collides with a cognee DataPoint.

The schema title becomes the generated class name, which is persisted as the graph
node ``type``. cognee resolves nodes by type (tools, skills), so a user-supplied title
matching a first-party model would let the schema mint nodes indistinguishable from
cognee's own. This is the schema-boundary half of the tool hardening.
"""

import pytest

from cognee.shared.exceptions import ReservedGraphModelTitleError
from cognee.shared.graph_model_utils import (
    _reserved_datapoint_type_names,
    graph_schema_to_graph_model,
)


def _schema(title: str) -> dict:
    return {
        "title": title,
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }


def test_first_party_types_are_reserved():
    reserved = _reserved_datapoint_type_names()
    # The privileged types consumed by name at retrieval time must be reserved.
    assert {"DataPoint", "Tool", "Skill"} <= reserved


@pytest.mark.parametrize("title", ["Tool", "Skill", "DataPoint"])
def test_reserved_title_is_rejected(title):
    with pytest.raises(ReservedGraphModelTitleError) as exc_info:
        graph_schema_to_graph_model(_schema(title))
    assert exc_info.value.status_code == 400
    assert title in exc_info.value.message


def test_benign_domain_title_is_accepted():
    model = graph_schema_to_graph_model(_schema("Machine"))
    assert model.__name__ == "Machine"
    assert "name" in model.model_fields


def test_external_ref_check_still_runs_first():
    """The reserved-title check must not shadow the pre-existing $ref guard."""
    from cognee.shared.exceptions import ExternalSchemaReferenceError

    schema = _schema("Tool")
    schema["properties"]["other"] = {"$ref": "https://example.com/x.json"}
    with pytest.raises(ExternalSchemaReferenceError):
        graph_schema_to_graph_model(schema)
