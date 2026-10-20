"""Which vector-store collections a node owns, lexical-only ones included (SDK-397)."""

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils.vector_fields import (
    lexical_fields_of_type,
    node_vector_fields,
    vector_fields_of,
)
from cognee.modules.engine.models.CachedDocument import CachedDocument


class _BothKinds(DataPoint):
    text: str
    title: str
    metadata: dict = {"index_fields": ["text"], "lexical_fields": ["text", "title"]}


def test_metadata_lists_embedded_fields_first_then_lexical_only_ones():
    assert vector_fields_of(_BothKinds(text="a", title="b").metadata) == ["text", "title"]
    assert vector_fields_of(CachedDocument.model_fields["metadata"].default) == ["text"]
    assert vector_fields_of(None) == []


def test_a_type_name_resolves_to_its_lexical_fields():
    assert lexical_fields_of_type("CachedDocument") == ["text"]
    assert lexical_fields_of_type("_BothKinds") == ["text", "title"]
    assert lexical_fields_of_type("NoSuchType") == []
    assert lexical_fields_of_type(None) == []


def test_a_delete_snapshot_gains_the_lexical_only_collections():
    # A cached document embeds nothing: its only collection is the lexical one.
    assert node_vector_fields("CachedDocument", []) == ["text"]
    assert node_vector_fields("_BothKinds", ["text"]) == ["text", "title"]
    assert node_vector_fields("Entity", ["name"]) == ["name"]
    assert node_vector_fields(None, None) == []
