"""COGX 0.3 metadata: the OMP memory-object fields (draft v0.6) on export,
through the loader, on re-export, and read back from 0.2 archives.

Pure tests: synthetic ``get_graph_data()``-shaped graphs, no databases, no LLM.
"""

import hashlib
import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from cognee.modules.migration.cogx import (
    COGX_VERSION,
    OMP_PROTOCOL_VERSION,
    COGXArchiveWriter,
    COGXConfidence,
    COGXContributor,
    COGXDocument,
    COGXEntity,
    COGXEvidence,
    COGXFact,
    COGXLink,
    COGXMemory,
    COGXOrigin,
    COGXVersion,
    checksum_of,
    content_hash_of,
    header_from_graph_properties,
    normalize_legacy_fields,
    omp_header_of,
    parse_omp_header,
    read_archive,
    read_manifest,
    verify_checksum,
)
from cognee.modules.migration.export import (
    COGX_EXPORT_COVERAGE,
    Actor,
    EdgeSupport,
    ExportFacts,
    _write_cogx,
)
from cognee.modules.migration.loader import translate_records

ALICE = "11111111-1111-4111-8111-111111111111"
BERLIN = "22222222-2222-4222-8222-222222222222"
PARIS = "66666666-6666-4666-8666-666666666666"
CHUNK = "44444444-4444-4444-8444-444444444444"
CHUNK_2 = "77777777-7777-4777-8777-777777777777"
DOC = "88888888-8888-4888-8888-888888888888"
DOC_ORIGIN = {"system": "mem0", "id": "mem-7"}
TEXT = "Alice lives in Berlin."


def graph():
    nodes = [
        (
            ALICE,
            {
                "type": "Entity",
                "name": "Alice",
                "description": "A person",
                "version": 3,
                "tags": json.dumps(["person", "team"]),
                "source_user": "user-1",
                "source_pipeline": "cognify_pipeline",
            },
        ),
        (
            BERLIN,
            {
                "type": "Entity",
                "name": "Berlin",
                "description": "A city",
                # Imported with an earlier origin: a cognee export keeps it.
                "omp_header": json.dumps({"origin": {"system": "mem0", "id": "mem-42"}}),
                "semantic_kind": "fact",
                "valid_from": "2026-01-01T00:00:00+00:00",
            },
        ),
        (PARIS, {"type": "Entity", "name": "Paris", "description": "A city"}),
        (CHUNK, {"type": "DocumentChunk", "text": TEXT, "chunk_index": 0, "source_user": "user-1"}),
        (
            CHUNK_2,
            {
                "type": "DocumentChunk",
                "text": "Paris is lovely.",
                "chunk_index": 0,
                "source_user": "agent-1",
            },
        ),
        # The data item CHUNK_2 was cognified from, imported from Mem0.
        (
            DOC,
            {
                "type": "TextDocument",
                "name": "memory",
                "external_metadata": json.dumps(
                    {"external_id": "mem-7", "omp_header": {"origin": DOC_ORIGIN}}
                ),
            },
        ),
    ]
    edges = [
        (CHUNK, ALICE, "contains", {}),
        (CHUNK, BERLIN, "contains", {}),
        (CHUNK_2, PARIS, "contains", {}),
        (CHUNK_2, DOC, "is_part_of", {}),
        (ALICE, BERLIN, "lives_in", {"edge_text": "Alice lives in Berlin", "confidence": 0.8}),
        # Conflict review found this fact contradicted by another source.
        (ALICE, PARIS, "visited", {"conflict_marks": [{"status": "conflicting"}]}),
    ]
    return nodes, edges


# What export_dataset loads outside the graph for it: recorded edge evidence
# and the users the source_user stamps name. ``visited`` has no evidence.
FACTS = ExportFacts(
    edge_support={
        f"{ALICE}:lives_in:{BERLIN}": [EdgeSupport(CHUNK, "extracted", 0.8)],
        f"{CHUNK}:contains:{ALICE}": [EdgeSupport(CHUNK, "structural", None)],
    },
    actors={
        "user-1": Actor(kind="human", accountable="user-1"),
        "agent-1": Actor(kind="agent", accountable="owner@example.com"),
    },
)


def export(tmp_path, origin=None):
    nodes, edges = graph()
    destination = tmp_path / "archive_cogx"
    _write_cogx(nodes, edges, destination, "main_dataset", FACTS, origin=origin)
    records = list(read_archive(destination))
    return destination, records


def by_id(records, kind, external_id):
    return next(r for r in records if r.kind == kind and r.external_id == external_id)


class TestModels:
    def test_body_hash_is_stamped_and_stable(self):
        document = COGXDocument(external_id="d", content=TEXT)
        assert document.content_hash == "sha256:" + hashlib.sha256(TEXT.encode()).hexdigest()
        assert content_hash_of(TEXT) == document.content_hash
        # Metadata differences do not change the body hash: that is what dedupe needs.
        other = COGXDocument(external_id="other", content=TEXT, subject=["Alice"])
        assert other.content_hash == document.content_hash
        # Entities render a body too, so every record kind has a hash and a media type.
        entity = COGXEntity(external_id="e", name="Alice", description="A person")
        assert entity.content_hash == content_hash_of("Alice: A person")
        assert entity.media_type == "text/plain"

    def test_an_empty_field_added_later_keeps_the_checksum(self):
        class LaterMemory(COGXMemory):
            later: list[str] = []  # a field a future COGX version adds

        record = COGXMemory(external_id="m", content=TEXT)
        assert checksum_of(LaterMemory(external_id="m", content=TEXT)) == checksum_of(record)

    def test_a_nested_empty_field_added_later_keeps_the_checksum(self):
        class LaterLink(COGXLink):
            later: list[str] = []  # a field a future COGX version adds to links

        class Before(BaseModel):
            links: list[COGXLink] = []

        class After(BaseModel):
            links: list[LaterLink] = []

        link = {"target": "t", "rel": "relates_to"}
        assert checksum_of(After(links=[link])) == checksum_of(Before(links=[link]))

    def test_a_sub_model_added_later_with_only_empty_fields_keeps_the_checksum(self):
        class Inner(BaseModel):
            later: list[str] = []

        class Before(BaseModel):
            name: str = "x"

        class After(Before):
            inner: Inner = Inner()  # dumps as {"later": []}, empty once cleaned

        assert checksum_of(After()) == checksum_of(Before())

    def test_legacy_float_confidence_becomes_score(self):
        fact = COGXFact(
            external_id="f", subject_ref="a", predicate="p", object_ref="b", confidence=0.8
        )
        assert fact.confidence == COGXConfidence(score=0.8)
        assert fact.confidence.outcome == "unverified"

    def test_fact_validity_aliases_sync_both_ways(self):
        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        old_style = COGXFact(
            external_id="f", subject_ref="a", predicate="p", object_ref="b", valid_at=when
        )
        assert old_style.valid_from == when
        new_style = COGXFact(
            external_id="f", subject_ref="a", predicate="p", object_ref="b", valid_until=when
        )
        assert new_style.invalid_at == when

    def test_version_declares_a_comparator(self):
        """OMP 3.3.2: an opaque value with a declared comparator; ints and
        strings are read as integer / opaque versions."""
        assert COGXVersion.model_validate(3) == COGXVersion(value=3, comparator="integer")
        assert COGXVersion.model_validate("rev-a").comparator == "opaque"
        assert COGXVersion(value=2).compare(COGXVersion(value=3)) == -1
        assert (
            COGXVersion(value="1.10.0", comparator="semver").compare(
                COGXVersion(value="1.9.0", comparator="semver")
            )
            == 1
        )
        # Different comparators, or opaque ones, are incomparable: None, never a guess.
        assert COGXVersion(value=1).compare(COGXVersion(value="1", comparator="opaque")) is None
        assert (
            COGXVersion.model_validate({"value": "b", "comparator": "opaque"}).compare(
                COGXVersion(value="a", comparator="opaque")
            )
            is None
        )
        with pytest.raises(ValueError):
            COGXVersion.model_validate(True)

    def test_header_survives_storage_roundtrip(self):
        record = COGXEntity(
            external_id="e",
            name="Alice",
            origin=COGXOrigin(system="mem0", id="m-1"),
            subject=["Alice"],
            semantic_kind="preference",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            invalidated_by="e-2",
            evidence=[COGXEvidence(relation="stated", ref="user:1", confidence=0.5)],
            links=[COGXLink(target="e-0", rel="derived_from")],
            confidence=COGXConfidence(outcome="confirmed", verified_by="human"),
            weight=1.5,
            path="people/alice.md",
            scope_tags=["team:platform"],
            extensions={"x_rank": 7},
        )
        stored = omp_header_of(record)
        # Per-record fields are recomputed on every export, so they are not stored.
        assert not {"content_hash", "media_type", "checksum", "protocol_version"} & set(stored)
        back = parse_omp_header(json.dumps(stored))  # stored as one JSON property
        assert back["origin"] == record.origin
        assert back["subject"] == ["Alice"]
        assert back["semantic_kind"] == "preference"
        assert back["valid_from"] == record.valid_from
        assert back["invalidated_by"] == "e-2"
        assert back["evidence"] == record.evidence
        assert back["links"] == record.links
        assert back["confidence"] == record.confidence
        assert back["weight"] == 1.5
        assert back["path"] == "people/alice.md"
        assert back["scope_tags"] == ["team:platform"]
        assert back["extensions"] == {"x_rank": 7}  # unknown fields preserved (OMP 1.3)

    def test_non_default_media_type_is_stored(self):
        """text/plain is what every body gets anyway; anything else is remembered."""
        assert "media_type" not in omp_header_of(COGXDocument(external_id="d", content="x"))
        stored = omp_header_of(
            COGXDocument(external_id="d", content="# x", mime_type="text/markdown")
        )
        assert stored["media_type"] == "text/markdown"
        assert parse_omp_header(stored)["media_type"] == "text/markdown"

    def test_malformed_header_fields_are_dropped_not_guessed(self):
        header = parse_omp_header(
            {
                "origin": "not an object",
                "subject": "not a list",
                "semantic_kind": "vibe",
                "evidence": [1],
                "confidence": "high",
                "tags": ["kept"],
            }
        )
        assert header == {"tags": ["kept"]}  # each field stands or falls alone
        assert parse_omp_header("[broken") == {}
        assert parse_omp_header(None) == {}
        # The numeric edge confidence retrieval reads becomes the measured score.
        assert header_from_graph_properties({"confidence": 0.8})["confidence"] == COGXConfidence(
            score=0.8
        )
        assert header_from_graph_properties({"valid_to": "2026-01-01T00:00:00Z"})[
            "valid_until"
        ] == datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestLegacyNames:
    """COGX 0.2 wrote the Oct-14 header names; 0.3 reads them unchanged."""

    def test_record_fields_are_mapped_on_read(self):
        row = {
            "kind": "document",
            "external_id": "d",
            "content": TEXT,
            "content_type": "text/markdown",
            "valid_to": "2026-02-01T00:00:00Z",
            "contributors": [
                {"role": "initiator", "actor": "user-1", "kind": "human"},
                {"role": "author", "actor": "model-x", "kind": "agent"},
                {"role": "approver", "actor": "user-2", "kind": "human"},
                {"role": "system", "actor": "cognee:cognify_pipeline", "kind": "agent"},
            ],
            "evidence": [{"role": "stated", "ref": "user:1", "confidence": 0.5}],
            "ai_involvement": "mixed",
        }
        record = COGXDocument.model_validate(row)
        assert record.media_type == "text/markdown"
        assert record.valid_until == datetime(2026, 2, 1, tzinfo=timezone.utc)
        assert [c.contribution for c in record.provenance] == [
            "instructed",
            "authored",
            "approved",
            "system",  # no OMP synonym: kept as written, not guessed
        ]
        assert record.provenance[0].id == "user-1"
        assert record.evidence == [COGXEvidence(ref="user:1", relation="stated", confidence=0.5)]
        assert record.ai_involvement == "mixed"

    def test_new_names_win_over_legacy_spellings(self):
        record = COGXDocument.model_validate(
            {
                "external_id": "d",
                "content": "x",
                "content_type": "text/csv",
                "media_type": "text/html",
            }
        )
        assert record.media_type == "text/html"

    def test_fact_provenance_strings_become_evidence(self):
        """COGX 0.1 facts listed their source episodes under ``provenance``:
        that is evidence, with no relation guessed."""
        fact = COGXFact.model_validate(
            {
                "external_id": "f",
                "subject_ref": "a",
                "predicate": "p",
                "object_ref": "b",
                "provenance": ["ep1", "ep2"],
            }
        )
        assert fact.provenance == []
        assert fact.evidence == [COGXEvidence(ref="ep1"), COGXEvidence(ref="ep2")]
        assert normalize_legacy_fields({"provenance": ["x"], "evidence": [{"ref": "y"}]})[
            "evidence"
        ] == [{"ref": "y"}, {"ref": "x"}]

    def test_fact_keeps_contributors_beside_provenance_strings(self):
        """A 0.2 fact carried episode refs under ``provenance`` and its
        authors under ``contributors``: neither is lost on read."""
        fact = COGXFact.model_validate(
            {
                "external_id": "f",
                "subject_ref": "a",
                "predicate": "p",
                "object_ref": "b",
                "provenance": ["ep1"],
                "contributors": [{"role": "author", "actor": "user-a", "kind": "human"}],
            }
        )
        assert fact.evidence == [COGXEvidence(ref="ep1")]
        assert [(c.id, c.contribution) for c in fact.provenance] == [("user-a", "authored")]

    def test_stored_0_2_header_is_read(self):
        stored = json.dumps(
            {
                "origin": {"system": "mem0", "id": "m-1"},
                "content_type": "text/markdown",
                "valid_to": "2026-02-01T00:00:00Z",
                "contributors": [{"role": "approver", "actor": "misha", "kind": "human"}],
                "evidence": [{"role": "derived", "ref": "m-0"}],
                "version": 2,
            }
        )
        header = parse_omp_header(stored)
        assert header["media_type"] == "text/markdown"
        assert header["valid_until"] == datetime(2026, 2, 1, tzinfo=timezone.utc)
        assert header["provenance"] == [
            COGXContributor(id="misha", kind="human", contribution="approved")
        ]
        assert header["evidence"] == [COGXEvidence(ref="m-0", relation="derived")]
        assert header["version"] == COGXVersion(value=2, comparator="integer")


class TestExportHeader:
    def test_manifest_declares_version_profiles_and_coverage(self, tmp_path):
        destination, _ = export(tmp_path, origin="cognee:ds-1")
        manifest = read_manifest(destination)
        assert manifest.cogx_version == COGX_VERSION == "0.3"
        assert manifest.protocol_version == OMP_PROTOCOL_VERSION == "0.6"
        assert manifest.origin == "cognee:ds-1"
        assert manifest.selection == "snapshot" and manifest.intent == "copy"
        assert manifest.media_types == ["text/plain"]
        # Every record renders as a conforming memory; not every one names a
        # human contributor and a version, so the audit profile is not claimed.
        assert manifest.profiles == ["minimal"]
        assert manifest.coverage == COGX_EXPORT_COVERAGE
        assert {"provenance", "accountable", "checksum", "links", "weight"} <= set(
            manifest.coverage
        )

    def test_audit_profile_claimed_only_when_every_record_meets_it(self, tmp_path):
        audited = COGXMemory(
            external_id="m-1",
            content="Likes tea",
            version=1,
            provenance=[COGXContributor(id="sam", kind="human", contribution="authored")],
        )
        with COGXArchiveWriter(tmp_path / "audit_cogx") as writer:
            writer.write(audited)
        assert read_manifest(tmp_path / "audit_cogx").profiles == ["minimal", "audit"]
        with COGXArchiveWriter(tmp_path / "mixed_cogx") as writer:
            writer.write(audited)
            writer.write(COGXMemory(external_id="m-2", content="Likes coffee"))
        assert read_manifest(tmp_path / "mixed_cogx").profiles == ["minimal"]
        with pytest.raises(ValueError, match="intent"):
            COGXArchiveWriter(tmp_path / "bad_cogx", intent="teleport")

    def test_legacy_0_1_archive_still_reads(self, tmp_path):
        destination, _ = export(tmp_path)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cogx_version"] = "0.1"
        for key in ("coverage", "protocol_version", "profiles", "media_types", "origin"):
            manifest.pop(key)
        manifest_path.write_text(json.dumps(manifest))
        assert read_manifest(destination).coverage == {}
        assert read_manifest(destination).profiles == []
        # 0.1 records have none of the header fields; they parse untouched.
        for line in (destination / "entities.jsonl").read_text().splitlines():
            row = json.loads(line)
            for key in ("origin", "subject", "evidence", "provenance"):
                row.pop(key, None)
            assert COGXEntity.model_validate(row).origin is None

    def test_origin_defaults_to_cognee_and_keeps_earlier_origin(self, tmp_path):
        _, records = export(tmp_path)
        assert by_id(records, "entity", ALICE).origin == COGXOrigin(system="cognee", id=ALICE)
        assert by_id(records, "entity", BERLIN).origin == COGXOrigin(system="mem0", id="mem-42")
        fact = by_id(records, "fact", f"{ALICE}:lives_in:{BERLIN}")
        assert fact.origin == COGXOrigin(system="cognee", id=fact.external_id)

    def test_chunk_takes_its_documents_imported_header(self, tmp_path):
        _, records = export(tmp_path)
        assert by_id(records, "document", CHUNK_2).origin == COGXOrigin(**DOC_ORIGIN)
        assert by_id(records, "document", CHUNK).origin == COGXOrigin(system="cognee", id=CHUNK)

    def test_document_hash_subject_and_entity_evidence(self, tmp_path):
        _, records = export(tmp_path)
        document = by_id(records, "document", CHUNK)
        assert document.content_hash == content_hash_of(TEXT)
        assert document.subject == ["Alice", "Berlin"]
        alice = by_id(records, "entity", ALICE)
        assert alice.subject == ["Alice"]
        assert alice.evidence == [COGXEvidence(relation="derived", ref=CHUNK)]

    def test_fact_evidence_is_the_recorded_support(self, tmp_path):
        _, records = export(tmp_path)
        lives_in = by_id(records, "fact", f"{ALICE}:lives_in:{BERLIN}")
        assert lives_in.evidence == [COGXEvidence(relation="inferred", ref=CHUNK, confidence=0.8)]
        assert lives_in.ai_involvement == "full"
        assert lives_in.confidence == COGXConfidence(score=0.8)
        assert lives_in.subject == ["Alice", "Berlin"]  # names, as documents and entities
        # Nothing recorded: no evidence and no AI claim, never a guess from co-mentions.
        visited = by_id(records, "fact", f"{ALICE}:visited:{PARIS}")
        assert visited.evidence == []
        assert visited.ai_involvement is None
        contains = by_id(records, "fact", f"{CHUNK}:contains:{ALICE}")
        assert contains.evidence == [COGXEvidence(relation="observed", ref=CHUNK)]
        assert contains.ai_involvement == "none"  # built without a model

    def test_node_properties_feed_kind_and_validity(self, tmp_path):
        _, records = export(tmp_path)
        berlin = by_id(records, "entity", BERLIN)
        assert berlin.semantic_kind == "fact"
        assert berlin.valid_from == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert by_id(records, "entity", ALICE).semantic_kind is None

    def test_conflict_review_sets_fact_confidence(self, tmp_path):
        _, records = export(tmp_path)
        visited = by_id(records, "fact", f"{ALICE}:visited:{PARIS}")
        assert visited.confidence == COGXConfidence(outcome="contradicted", verified_by="automatic")
        # Absence means "not stated", never "confirmed".
        assert by_id(records, "entity", ALICE).confidence is None

    def test_review_verdict_wins_over_imported_confidence(self, tmp_path):
        nodes = [(ALICE, {"type": "Entity", "name": "Alice"})]
        imported = json.dumps({"confidence": {"outcome": "confirmed", "verified_by": "agent"}})
        edges = [
            (
                ALICE,
                ALICE,
                "current",
                {"omp_header": imported, "conflict_marks": [{"status": "current"}]},
            ),
            (
                ALICE,
                ALICE,
                "chosen",
                {
                    "omp_header": imported,
                    "conflict_marks": [{"status": "current", "resolution_source": "user"}],
                },
            ),
        ]
        destination = tmp_path / "reviewed_cogx"
        _write_cogx(nodes, edges, destination, "main_dataset", FACTS)
        records = list(read_archive(destination))
        current = by_id(records, "fact", f"{ALICE}:current:{ALICE}")
        assert current.confidence == COGXConfidence(outcome="confirmed", verified_by="automatic")
        chosen = by_id(records, "fact", f"{ALICE}:chosen:{ALICE}")
        assert chosen.confidence == COGXConfidence(outcome="confirmed", verified_by="human")


class TestLoaderRetention:
    def test_header_lands_in_metadata_and_edge_properties(self, tmp_path):
        _, records = export(tmp_path)
        result = translate_records(records, "preserve")
        document = next(item for item in result.data_items if TEXT in item.data)
        header = document.external_metadata["omp_header"]
        assert "content_hash" not in header  # recomputed by every export
        assert header["subject"] == ["Alice", "Berlin"]
        assert header["origin"] == {"system": "cognee", "id": CHUNK}

        edges = [edge for batch in result.graph_batches for edge in batch["edges"]]
        lives_in = next(edge for edge in edges if edge[2] == "lives_in")
        properties = lives_in[3]
        stored = json.loads(properties["omp_header"])  # one JSON property
        assert stored["evidence"] == [{"relation": "inferred", "ref": CHUNK, "confidence": 0.8}]
        assert stored["subject"] == ["Alice", "Berlin"]
        assert stored["origin"] == {"system": "cognee", "id": f"{ALICE}:lives_in:{BERLIN}"}
        assert properties["confidence"] == 0.8  # numeric, as retrieval expects

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
        no_facts = ExportFacts(edge_support={}, actors={})  # a fresh target recorded none
        _write_cogx(nodes, [(ALICE, BERLIN, "lives_in", lives_in[3])], second, "copy", no_facts)
        fact = by_id(list(read_archive(second)), "fact", f"{ALICE}:lives_in:{BERLIN}")
        assert fact.origin == COGXOrigin(system="cognee", id=f"{ALICE}:lives_in:{BERLIN}")
        assert fact.evidence == [COGXEvidence(relation="inferred", ref=CHUNK, confidence=0.8)]
        assert fact.valid_from is None

    def test_import_without_origin_originates_at_its_source(self):
        memory = COGXMemory(external_system="mem0", external_id="mem-42", content="Likes tea")
        result = translate_records([memory], "preserve")
        header = result.data_items[0].external_metadata["omp_header"]
        assert header["origin"] == {"system": "mem0", "id": "mem-42"}
        # A record that does not say where it comes from gets no invented origin.
        anonymous = COGXMemory(external_id="m-1", content="Likes tea")
        assert "origin" not in omp_header_of(anonymous)

    def test_entity_header_survives_import_and_reexport(self, tmp_path):
        record = COGXEntity(
            external_system="mem0",
            external_id="e-1",
            name="Alice",
            origin=COGXOrigin(system="mem0", id="e-1"),
            tags=["person"],
            links=[COGXLink(target="e-0", rel="supersedes", version=1)],
            weight=0.4,
        )
        result = translate_records([record], "preserve")
        entity = next(
            node
            for batch in result.graph_batches
            for node in batch["nodes"]
            if getattr(node, "name", None) == "Alice"
        )
        nodes = [
            (str(entity.id), {"type": "Entity", "name": "Alice", "omp_header": entity.omp_header})
        ]
        destination = tmp_path / "entity_cogx"
        _write_cogx(nodes, [], destination, "copy", ExportFacts(edge_support={}, actors={}))
        exported = by_id(list(read_archive(destination)), "entity", str(entity.id))
        assert exported.origin == COGXOrigin(system="mem0", id="e-1")
        assert exported.tags == ["person"]
        assert exported.links == [COGXLink(target="e-0", rel="supersedes", version=1)]
        assert exported.weight == 0.4


class TestRemainingHeaderRows:
    def test_writer_stamps_record_checksum_and_tamper_is_detected(self, tmp_path):
        _, records = export(tmp_path)
        document = by_id(records, "document", CHUNK)
        assert document.checksum and document.checksum.startswith("sha256:")
        assert verify_checksum(document) is True
        assert document.checksum != document.content_hash  # record hash is not the body hash
        document.subject = ["someone else"]
        assert verify_checksum(document) is False
        assert verify_checksum(COGXDocument(external_id="x", content="y")) is None

    def test_checksum_is_canonical(self):
        """Field order and absent-vs-empty fields never change the checksum."""
        one = COGXMemory(external_id="m", content="x", tags=["a"])
        two = COGXMemory.model_validate({"tags": ["a"], "content": "x", "external_id": "m"})
        assert checksum_of(one) == checksum_of(two)
        assert checksum_of(one) != checksum_of(COGXMemory(external_id="m", content="x"))

    def test_media_type_declares_body_format(self, tmp_path):
        assert (
            COGXDocument(external_id="d", content="# hi", mime_type="text/markdown").media_type
            == "text/markdown"
        )
        _, records = export(tmp_path)
        assert by_id(records, "document", CHUNK).media_type == "text/plain"
        assert by_id(records, "fact", f"{ALICE}:lives_in:{BERLIN}").media_type == "text/plain"

    def test_version_and_tags_from_node_properties(self, tmp_path):
        _, records = export(tmp_path)
        alice = by_id(records, "entity", ALICE)
        assert alice.version == COGXVersion(value=3, comparator="integer")
        assert alice.tags == ["person", "team"]
        assert by_id(records, "entity", PARIS).version is None

    def test_provenance_names_accountable_human_and_ai_involvement(self, tmp_path):
        _, records = export(tmp_path)
        alice = by_id(records, "entity", ALICE)
        assert alice.accountable == "user-1"
        assert alice.ai_involvement == "full"  # extracted by the pipeline
        assert alice.provenance == [
            COGXContributor(id="user-1", kind="human", contribution="instructed"),
            COGXContributor(
                id="cognee:cognify_pipeline",
                kind="agent",
                contribution="authored",  # a model wrote the record
                ai_usage="full",
            ),
        ]
        chunk = by_id(records, "document", CHUNK)
        assert chunk.ai_involvement == "none"  # a source artifact
        assert chunk.accountable == "user-1"
        # An agent wrote CHUNK_2: its owning human is the accountable one.
        by_agent = by_id(records, "document", CHUNK_2)
        assert by_agent.provenance[0] == COGXContributor(
            id="agent-1", kind="agent", contribution="instructed"
        )
        assert by_agent.accountable == "owner@example.com"
        # No stamps on the node: nothing is invented.
        paris = by_id(records, "entity", PARIS)
        assert paris.provenance == [] and paris.accountable is None
        assert paris.ai_involvement is None  # not extracted by cognify

    def test_pipeline_imports_what_no_model_wrote(self, tmp_path):
        nodes = [
            (
                CHUNK,
                {
                    "type": "DocumentChunk",
                    "text": TEXT,
                    "source_user": "user-1",
                    "source_pipeline": "cognify_pipeline",
                },
            )
        ]
        destination = tmp_path / "chunk_cogx"
        _write_cogx(nodes, [], destination, "main_dataset", FACTS)
        chunk = by_id(list(read_archive(destination)), "document", CHUNK)
        assert chunk.provenance[1] == COGXContributor(
            id="cognee:cognify_pipeline", kind="agent", contribution="imported", ai_usage="none"
        )

    def test_remaining_rows_survive_storage_roundtrip(self):
        record = COGXEntity(
            external_id="e",
            name="Alice",
            version=2,
            tags=["a", "b"],
            provenance=[COGXContributor(id="misha", kind="human", contribution="approved")],
            accountable="misha",
            ai_involvement="mixed",
        )
        back = parse_omp_header(omp_header_of(record))
        assert back["version"] == COGXVersion(value=2, comparator="integer")
        assert back["tags"] == ["a", "b"]
        assert back["provenance"] == record.provenance
        assert back["accountable"] == "misha"
        assert back["ai_involvement"] == "mixed"
        # Per-record fields describe one export's bytes and are never stored.
        assert "media_type" not in back and "checksum" not in back

    def test_loader_retains_remaining_rows(self, tmp_path):
        _, records = export(tmp_path)
        result = translate_records(records, "preserve")
        document = next(item for item in result.data_items if TEXT in item.data)
        header = document.external_metadata["omp_header"]
        assert "checksum" not in header and "media_type" not in header
        assert header["accountable"] == "user-1"
        assert header["ai_involvement"] == "none"
