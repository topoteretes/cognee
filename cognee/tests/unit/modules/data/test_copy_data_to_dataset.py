"""The pure parts of ``copy_data_to_dataset``: which refs a document owns, how they are
re-addressed to the destination, what never travels as a property, and which schema
copied vector rows are validated against."""

from types import ModuleType
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.provenance.source_refs import (
    make_chunk_source_ref_key,
    make_source_ref_key,
    parse_source_ref_key,
)
from cognee.modules.data.exceptions import VectorAdapterIndexSchemaError
from cognee.modules.data.methods.copy_data_to_dataset import (
    _index_schema_of,
    _refs_of_document,
    _rewrite_ref,
    _without_provenance,
)


def test_refs_of_document_keeps_only_the_documents_keys_of_shared_elements():
    dataset, mine, other = uuid4(), uuid4(), uuid4()
    chunk = uuid4()
    edge = EdgeIdentity("a", "b", "knows")
    refs = {
        "shared": [
            make_source_ref_key(dataset, other),
            make_chunk_source_ref_key(dataset, mine, chunk),
        ],
        "only_other": [make_source_ref_key(dataset, other)],
        edge: [make_source_ref_key(dataset, mine)],
    }

    selected = _refs_of_document(refs, mine)

    assert selected == {
        "shared": [make_chunk_source_ref_key(dataset, mine, chunk)],
        edge: [make_source_ref_key(dataset, mine)],
    }


def test_rewrite_ref_keeps_the_chunk_and_moves_dataset_and_data_id():
    source_dataset, source_data, target_dataset, target_data, chunk = (uuid4() for _ in range(5))

    whole = _rewrite_ref(
        make_source_ref_key(source_dataset, source_data), target_dataset, target_data
    )
    per_chunk = _rewrite_ref(
        make_chunk_source_ref_key(source_dataset, source_data, chunk), target_dataset, target_data
    )

    assert parse_source_ref_key(whole).dataset_id == target_dataset
    assert parse_source_ref_key(whole).data_id == target_data
    assert parse_source_ref_key(whole).chunk_id is None
    assert parse_source_ref_key(per_chunk).chunk_id == chunk
    assert parse_source_ref_key(per_chunk).data_id == target_data


def test_without_provenance_drops_exactly_the_graph_ownership_columns():
    properties = {
        "id": "n1",
        "name": "Anna",
        "source_ref_keys": "|x|",
        "source_dataset_ids": "|d|",
        "source_run_ids": "|r|",
        "source_run_refs": "|rr|",
        "description": "kept",
    }

    assert _without_provenance(properties) == {"id": "n1", "name": "Anna", "description": "kept"}


def test_index_schema_of_reads_the_adapters_module(monkeypatch):
    import sys

    module = ModuleType("fake_vector_adapter")
    module.IndexSchema = type("IndexSchema", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter_class = type("FakeAdapter", (), {"__module__": module.__name__})

    assert _index_schema_of(adapter_class()) is module.IndexSchema

    bare = ModuleType("bare_vector_adapter")
    monkeypatch.setitem(sys.modules, bare.__name__, bare)
    bare_class = type("BareAdapter", (), {"__module__": bare.__name__})
    with pytest.raises(VectorAdapterIndexSchemaError, match="BareAdapter"):
        _index_schema_of(bare_class())


def test_in_tree_adapters_expose_an_index_schema():
    pytest.importorskip("lancedb")
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    adapter = object.__new__(LanceDBAdapter)  # the module is what matters, not an instance
    assert _index_schema_of(adapter).__name__ == "IndexSchema"
