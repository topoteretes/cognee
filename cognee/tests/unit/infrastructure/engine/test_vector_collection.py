"""``DataPoint.vector_collection``: the name a type's index field is embedded under."""

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DltRow import DltRow
from cognee.modules.engine.models.DltColumn import DltColumn
from cognee.tasks.schema.models import SchemaRelationship, SchemaTable


class OneField(DataPoint):
    text: str = ""
    metadata: dict = {"index_fields": ["text"]}


class TwoFields(DataPoint):
    name: str = ""
    description: str = ""
    metadata: dict = {"index_fields": ["description", "name"]}


class NoFields(DataPoint):
    metadata: dict = {"index_fields": []}


def test_a_single_index_field_names_the_collection_from_the_model_alone():
    assert OneField.vector_collection() == "OneField_text"
    assert OneField.vector_collection("text") == "OneField_text"


def test_several_index_fields_must_be_named():
    with pytest.raises(ValueError, match=r"indexes \['description', 'name'\]; pass the field"):
        TwoFields.vector_collection()
    assert TwoFields.vector_collection("name") == "TwoFields_name"
    assert TwoFields.vector_collection("description") == "TwoFields_description"


def test_a_field_the_type_does_not_index_is_an_error():
    with pytest.raises(ValueError, match="does not index 'title'"):
        TwoFields.vector_collection("title")
    with pytest.raises(ValueError, match="indexes no fields"):
        NoFields.vector_collection()


def test_the_dlt_types_name_the_collections_the_pipeline_writes():
    assert DltRow.vector_collection() == "DltRow_text"
    assert DltColumn.vector_collection() == "DltColumn_properties"
    assert SchemaTable.vector_collection("name") == "SchemaTable_name"
    assert SchemaRelationship.vector_collection("name") == "SchemaRelationship_name"
    with pytest.raises(ValueError):
        SchemaTable.vector_collection()  # two index fields: name or description
