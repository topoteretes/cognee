"""COGX 0.2 header: the OMP fields (origin, content_hash, subject, kind,
validity, evidence, accuracy) on export, through the loader, and on re-export.

Pure tests: synthetic ``get_graph_data()``-shaped graphs, no databases, no LLM.
"""

import hashlib
import json
from datetime import datetime, timezone

from cognee.modules.migration.cogx import (
    COGX_VERSION,
    COGXAccuracy,
    COGXDocument,
    COGXEntity,
    COGXEvidence,
    COGXFact,
    COGXOrigin,
    content_hash_of,
    header_from_properties,
    header_to_properties,
    read_archive,
    read_manifest,
)
from cognee.modules.migration.export import COGX_EXPORT_COVERAGE, _write_cogx
from cognee.modules.migration.loader import translate_records

ALICE = "11111111-1111-4111-8111-111111111111"
BERLIN = "22222222-2222-4222-8222-222222222222"
PARIS = "66666666-6666-4666-8666-666666666666"
CHUNK = "44444444-4444-4444-8444-444444444444"
CHUNK_2 = "77777777-7777-4777-8777-777777777777"
TEXT = "Alice lives in Berlin."


def graph():
    nodes = [
        (ALICE, {"type": "Entity", "name": "Alice", "description": "A person"}),
        (
            BERLIN,
            {
                "type": "Entity",
                "name": "Berlin",
                "description": "A city",
                # An earlier origin survives a cognee export (copy keeps identity).
                "origin": json.dumps({"system": "mem0", "id": "mem-42"}),
                "semantic_kind": "fact",
                "valid_from": "2026-01-01T00:00:00+00:00",
            },
        ),
        (PARIS, {"type": "Entity", "name": "Paris", "description": "A city"}),
        (CHUNK, {"type": "DocumentChunk", "text": TEXT, "chunk_index": 0}),
        (CHUNK_2, {"type": "DocumentChunk", "text": "Paris is lovely.", "chunk_index": 1}),
    ]
    edges = [
        (CHUNK, ALICE, "contains", {}),
        (CHUNK, BERLIN, "contains", {}),
        (CHUNK_2, PARIS, "contains", {}),
        (ALICE, BERLIN, "lives_in", {"edge_text": "Alice lives in Berlin", "confidence": 0.8}),
        (ALICE, PARIS, "visited", {}),
        (BERLIN, PARIS, "contradicts", {}),
    ]
    return nodes, edges


def export(tmp_path):
    nodes, edges = graph()
    destination = tmp_path / "archive_cogx"
    _write_cogx(nodes, edges, destination, "main_dataset")
    records = list(read_archive(destination))
    return destination, records


def by_id(records, kind, external_id):
    return next(r for r in records if r.kind == kind and r.external_id == external_id)


class TestModels:
    def test_body_hash_is_stamped_and_stable(self):
        document = COGXDocument(external_id="d", content=TEXT)
        assert document.content_hash == "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest()
        assert content_hash_of(TEXT) == document.content_hash
        # Header differences do not change the body hash: that is what dedupe needs.
        other = COGXDocument(external_id="other", content=TEXT, subject=["Alice"])
        assert other.content_hash == document.content_hash
        assert COGXEntity(external_id="e", name="Alice").content_hash is None

    def test_fact_validity_aliases_sync_both_ways(self):
        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        old_style = COGXFact(
            external_id="f", subject_ref="a", predicate="p", object_ref="b", valid_at=when
        )
        assert old_style.valid_from == when
        new_style = COGXFact(
            external_id="f", subject_ref="a", predicate="p", object_ref="b", valid_to=when
        )
        assert new_style.invalid_at == when

    def test_header_survives_property_roundtrip(self):
        record = COGXEntity(
            external_id="e",
            name="Alice",
            origin=COGXOrigin(system="mem0", id="m-1"),
            subject=["Alice"],
            semantic_kind="preference",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            invalidated_by="e-2",
            evidence=[COGXEvidence(role="stated", ref="user:1", confidence=0.5)],
            accuracy=COGXAccuracy(outcome="confirmed", verified_by="human"),
        )
        properties = header_to_properties(record)
        # Flat and JSON-safe: nested values are strings a graph backend can store.
        assert all(isinstance(value, str) for value in properties.values())
        back = header_from_properties(properties)
        assert back["origin"] == record.origin
        assert back["subject"] == ["Alice"]
        assert back["semantic_kind"] == "preference"
        assert back["valid_from"] == record.valid_from
        assert back["invalidated_by"] == "e-2"
        assert back["evidence"] == record.evidence
        assert back["accuracy"] == record.accuracy

    def test_malformed_properties_are_dropped_not_guessed(self):
        header = header_from_properties(
            {
                "origin": "not json",
                "subject": "[broken",
                "semantic_kind": "vibe",
                "evidence": "[1]",
                "accuracy": "{}",
            }
        )
        assert "origin" not in header
        assert "subject" not in header
        assert "semantic_kind" not in header
        assert "evidence" not in header
        assert header["accuracy"] == COGXAccuracy()  # empty object: unverified


class TestExportHeader:
    def test_manifest_declares_version_and_coverage(self, tmp_path):
        destination, _ = export(tmp_path)
        manifest = read_manifest(destination)
        assert manifest.cogx_version == COGX_VERSION == "0.2"
        assert manifest.coverage == COGX_EXPORT_COVERAGE
        assert "provenance" in manifest.coverage

    def test_legacy_0_1_archive_still_reads(self, tmp_path):
        destination, _ = export(tmp_path)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cogx_version"] = "0.1"
        manifest.pop("coverage")
        manifest_path.write_text(json.dumps(manifest))
        assert read_manifest(destination).coverage == {}
        # 0.1 records have none of the header fields; they parse untouched.
        for line in (destination / "entities.jsonl").read_text().splitlines():
            row = json.loads(line)
            for key in ("origin", "subject", "evidence"):
                row.pop(key, None)
            assert COGXEntity.model_validate(row).origin is None

    def test_origin_defaults_to_cognee_and_keeps_earlier_origin(self, tmp_path):
        _, records = export(tmp_path)
        assert by_id(records, "entity", ALICE).origin == COGXOrigin(system="cognee", id=ALICE)
        assert by_id(records, "entity", BERLIN).origin == COGXOrigin(system="mem0", id="mem-42")
        fact = by_id(records, "fact", f"{ALICE}:lives_in:{BERLIN}")
        assert fact.origin == COGXOrigin(system="cognee", id=fact.external_id)

    def test_document_hash_subject_and_entity_evidence(self, tmp_path):
        _, records = export(tmp_path)
        document = by_id(records, "document", CHUNK)
        assert document.content_hash == content_hash_of(TEXT)
        assert document.subject == ["Alice", "Berlin"]
        alice = by_id(records, "entity", ALICE)
        assert alice.subject == ["Alice"]
        assert alice.evidence == [COGXEvidence(role="derived", ref=CHUNK)]

    def test_fact_evidence_prefers_chunks_mentioning_both_sides(self, tmp_path):
        _, records = export(tmp_path)
        lives_in = by_id(records, "fact", f"{ALICE}:lives_in:{BERLIN}")
        assert lives_in.evidence == [COGXEvidence(role="derived", ref=CHUNK, confidence=0.8)]
        assert lives_in.subject == [ALICE, BERLIN]
        # No chunk mentions both Alice and Paris: fall back to either side, in order.
        visited = by_id(records, "fact", f"{ALICE}:visited:{PARIS}")
        assert [item.ref for item in visited.evidence] == [CHUNK, CHUNK_2]
        # Structural edges carry no derived evidence.
        contains = by_id(records, "fact", f"{CHUNK}:contains:{ALICE}")
        assert contains.evidence == []

    def test_node_properties_feed_kind_and_validity(self, tmp_path):
        _, records = export(tmp_path)
        berlin = by_id(records, "entity", BERLIN)
        assert berlin.semantic_kind == "fact"
        assert berlin.valid_from == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert by_id(records, "entity", ALICE).semantic_kind is None

    def test_contradicts_edges_mark_accuracy(self, tmp_path):
        _, records = export(tmp_path)
        for node_id in (BERLIN, PARIS):
            accuracy = by_id(records, "entity", node_id).accuracy
            assert accuracy.outcome == "contradicted"
            assert accuracy.verified_by == "automatic"
        # Absence means "not stated", never "confirmed".
        assert by_id(records, "entity", ALICE).accuracy is None


class TestLoaderRetention:
    def test_header_lands_in_metadata_and_edge_properties(self, tmp_path):
        _, records = export(tmp_path)
        result = translate_records(records, "preserve")
        document = next(item for item in result.data_items if TEXT in item.data)
        header = document.external_metadata["header"]
        assert header["content_hash"] == content_hash_of(TEXT)
        assert header["subject"] == ["Alice", "Berlin"]
        assert header["origin"] == {"system": "cognee", "id": CHUNK}

        edges = [edge for batch in result.graph_batches for edge in batch["edges"]]
        lives_in = next(edge for edge in edges if edge[2] == "lives_in")
        properties = lives_in[3]
        assert json.loads(properties["evidence"]) == [
            {"role": "derived", "ref": CHUNK, "confidence": 0.8}
        ]
        assert json.loads(properties["subject"]) == [ALICE, BERLIN]
        assert json.loads(properties["origin"]) == {
            "system": "cognee",
            "id": f"{ALICE}:lives_in:{BERLIN}",
        }

    def test_reexport_keeps_origin_from_imported_edge(self, tmp_path):
        """An edge restored from an archive re-exports with the origin it arrived with."""
        _, records = export(tmp_path)
        result = translate_records(records, "preserve")
        edges = [edge for batch in result.graph_batches for edge in batch["edges"]]
        lives_in = next(edge for edge in edges if edge[2] == "lives_in")
        nodes = [
            (ALICE, {"type": "Entity", "name": "Alice"}),
            (BERLIN, {"type": "Entity", "name": "Berlin"}),
        ]
        second = tmp_path / "second_cogx"
        _write_cogx(nodes, [(ALICE, BERLIN, "lives_in", lives_in[3])], second, "copy")
        fact = by_id(list(read_archive(second)), "fact", f"{ALICE}:lives_in:{BERLIN}")
        assert fact.origin == COGXOrigin(system="cognee", id=f"{ALICE}:lives_in:{BERLIN}")
        assert fact.evidence == [COGXEvidence(role="derived", ref=CHUNK, confidence=0.8)]
        assert fact.valid_from is None
