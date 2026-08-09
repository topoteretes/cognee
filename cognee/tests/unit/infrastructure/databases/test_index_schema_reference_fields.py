"""Regression tests: every vector adapter's IndexSchema must carry the
reference scalars used by the search "Evidence" feature.

These are pure schema/serialization checks (no DB connection) guarding against
an adapter silently dropping ``document_id`` / ``document_name`` / ``chunk_index``
from its payload — which would make chunk Evidence render empty on that backend.
"""

import pytest

from cognee.modules.graph.models.EdgeInstance import EdgeInstance

REFERENCE_FIELDS = ("document_id", "document_name", "chunk_index", "source_chunk_id")


def _index_schema_classes():
    # LanceDB is the default vector backend and is always installed.
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema as LanceDBIndexSchema,
    )

    classes = [("lancedb", LanceDBIndexSchema)]

    # PGVector and Neptune require optional extras (postgres / aws). Importing
    # their adapters fails when those deps are absent (e.g. the basic CI job),
    # so include them only when importable — they are covered in the dedicated
    # DB test jobs where the extras are installed.
    try:
        from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
            IndexSchema as PGVectorIndexSchema,
        )

        classes.append(("pgvector", PGVectorIndexSchema))
    except Exception:  # pragma: no cover - depends on optional extras
        pass

    try:
        from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
            IndexSchema as NeptuneIndexSchema,
        )

        classes.append(("neptune", NeptuneIndexSchema))
    except Exception:  # pragma: no cover - depends on optional extras
        pass

    return classes


@pytest.mark.parametrize("name, schema_cls", _index_schema_classes())
def test_index_schema_defaults_reference_fields_to_none(name, schema_cls):
    # Non-chunk data points omit these fields; they must default to None,
    # never raise, so the schema stays compatible with every DataPoint type.
    instance = schema_cls(id="11111111-1111-1111-1111-111111111111", text="hello")
    dumped = instance.model_dump()
    for field in REFERENCE_FIELDS:
        assert field in dumped, f"{name} IndexSchema is missing {field}"
        assert dumped[field] is None


@pytest.mark.parametrize("name, schema_cls", _index_schema_classes())
def test_index_schema_round_trips_reference_fields(name, schema_cls):
    instance = schema_cls(
        id="11111111-1111-1111-1111-111111111111",
        text="some chunk text",
        document_id="22222222-2222-2222-2222-222222222222",
        document_name="annual_report.pdf",
        chunk_index=4,
        source_chunk_id="33333333-3333-3333-3333-333333333333",
        importance_weight=0.8,
    )
    dumped = instance.model_dump()
    assert dumped["document_id"] == "22222222-2222-2222-2222-222222222222"
    assert dumped["document_name"] == "annual_report.pdf"
    assert dumped["chunk_index"] == 4
    assert dumped["source_chunk_id"] == "33333333-3333-3333-3333-333333333333"
    assert dumped["importance_weight"] == 0.8


def test_edge_payload_fields_survive_index_schema_conversion():
    """Edge-instance context must remain searchable after vector conversion.

    Removing any one of the edge fields from the common index payload would
    make this regression fail.
    """
    from cognee.infrastructure.databases.vector.models.IndexSchema import (
        index_schema_from_data_point,
    )

    point = EdgeInstance(
        id="15bfc0f0-51d7-5ac8-8589-3c32fe75aa10",
        text="A depends on B.",
        relationship_name="depends_on",
        source_node_id="a",
        target_node_id="b",
    )

    indexed = index_schema_from_data_point(point)

    assert indexed.model_dump()["relationship_name"] == "depends_on"
    assert indexed.model_dump()["source_node_id"] == "a"
    assert indexed.model_dump()["target_node_id"] == "b"
