"""Regression tests: every vector adapter's IndexSchema must carry the
reference scalars used by the search "Evidence" feature.

These are pure schema/serialization checks (no DB connection) guarding against
an adapter silently dropping ``document_id`` / ``document_name`` / ``chunk_index``
from its payload — which would make chunk Evidence render empty on that backend.
"""

import pytest

REFERENCE_FIELDS = ("document_id", "document_name", "chunk_index")


def _index_schema_classes():
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        IndexSchema as LanceDBIndexSchema,
    )
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        IndexSchema as PGVectorIndexSchema,
    )
    from cognee.infrastructure.databases.hybrid.neptune_analytics.NeptuneAnalyticsAdapter import (
        IndexSchema as NeptuneIndexSchema,
    )

    return [
        ("lancedb", LanceDBIndexSchema),
        ("pgvector", PGVectorIndexSchema),
        ("neptune", NeptuneIndexSchema),
    ]


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
    )
    dumped = instance.model_dump()
    assert dumped["document_id"] == "22222222-2222-2222-2222-222222222222"
    assert dumped["document_name"] == "annual_report.pdf"
    assert dumped["chunk_index"] == 4
