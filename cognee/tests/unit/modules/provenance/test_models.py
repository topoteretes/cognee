from cognee.modules.provenance.models import ProvenanceEdgeEvidence


def test_edge_evidence_schema_stays_compact_and_delete_independent():
    table = ProvenanceEdgeEvidence.__table__

    assert {index.name for index in table.indexes} == {
        "ix_prov_evidence_edge",
        "ix_prov_evidence_source",
        "ix_prov_evidence_run",
    }
    assert not table.foreign_keys
