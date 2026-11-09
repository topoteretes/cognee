"""The OMP memory object (editors' draft v0.6): validation, checksum, the
conformance profiles, the runtime projection, the markdown carrier, the
mapping to and from COGX records, the ``omp`` exports and the ``OMPSource``
import with its receipt.

Pure tests: no databases, no LLM, no network.
"""

import asyncio
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cognee.modules.engine.models import Entity
from cognee.modules.migration.cogx import (
    COGXContributor,
    COGXDocument,
    COGXEntity,
    COGXEpisode,
    COGXEvidence,
    COGXFact,
    COGXLink,
    COGXMemory,
    COGXMemoryBlock,
    COGXScope,
    COGXTurn,
    COGXVersion,
)
from cognee.modules.migration.export import ExportFacts, _write_omp
from cognee.modules.migration.loader import translate_records
from cognee.modules.migration.omp import (
    OMP_INDEX_FILE,
    OMP_MEMORIES_FILE,
    OMPMemory,
    iter_omp_memories,
    mark_dangling,
    memory_from_record,
    read_omp_manifest,
    record_from_memory,
    write_omp_export,
)
from cognee.modules.migration.receipt import MAX_RECEIPT_ENTRIES, ImportReceipt
from cognee.modules.migration.sources.omp import OMPSource

# Appendix A of the draft: one memory in interchange form, every audit field.
APPENDIX_A = {
    "identifier": "9b2e7c14-3f6a-4d1e-8a5b-2c7f9e0d4b31",
    "protocol_version": "0.6",
    "path": "team/preferences/status-update-day.md",
    "title": "Status updates move to Friday",
    "media_type": "text/markdown",
    "tags": ["preference", "team-process"],
    "scope": ["team:platform"],
    "created_at": "2026-09-15T09:30:00Z",
    "valid_from": "2026-09-15T00:00:00Z",
    "version": {"value": 2, "comparator": "integer"},
    "checksum": "sha256:9f2c1e0b6d4a8f3e2b1c7d5a9e0f4b3c8d2a6e1f7b9c3d5e8a0f2b4c6d1e3a5",
    "provenance": [
        {
            "id": "user:jordan",
            "kind": "human",
            "contribution": "instructed",
            "at": "2026-09-15T09:28:00Z",
        },
        {
            "id": "agent:notes-assistant",
            "kind": "agent",
            "contribution": "authored",
            "name": "example-model-1",
        },
        {
            "id": "user:sam",
            "kind": "human",
            "contribution": "approved",
            "at": "2026-09-15T09:30:00Z",
        },
        {
            "id": "agent:review-judge",
            "kind": "agent",
            "contribution": "verified",
            "name": "example-judge-1",
            "at": "2026-10-01T12:00:00Z",
        },
    ],
    "evidence": [
        {"ref": "transcripts/2026-09-15-standup#turn-14", "relation": "stated"},
        {"ref": "https://wiki.example.com/team/process", "relation": "observed", "dangling": True},
    ],
    "links": [
        {"target": "9b2e7c14-3f6a-4d1e-8a5b-2c7f9e0d4b31", "version": 1, "rel": "supersedes"}
    ],
    "weight": 1.7,
    "body": "Status updates are due Fridays from Sep 15. They were due Tuesdays before that.",
}
WHEN = datetime(2026, 10, 14, tzinfo=timezone.utc)

ALICE = "11111111-1111-4111-8111-111111111111"
BERLIN = "22222222-2222-4222-8222-222222222222"
CHUNK = "44444444-4444-4444-8444-444444444444"
DOC = "88888888-8888-4888-8888-888888888888"
TEXT = "Alice lives in Berlin."
NO_FACTS = ExportFacts(edge_support={}, actors={})


def appendix_a() -> OMPMemory:
    return OMPMemory.model_validate(APPENDIX_A)


def graph():
    nodes = [
        (ALICE, {"type": "Entity", "name": "Alice", "description": "A person", "version": 1}),
        (BERLIN, {"type": "Entity", "name": "Berlin", "description": "A city"}),
        (CHUNK, {"type": "DocumentChunk", "text": TEXT, "chunk_index": 0}),
        (DOC, {"type": "TextDocument", "name": "memory"}),  # structure, not a memory
    ]
    edges = [
        (CHUNK, ALICE, "contains", {}),
        (CHUNK, BERLIN, "contains", {}),
        (CHUNK, DOC, "is_part_of", {}),
        (ALICE, BERLIN, "lives_in", {"edge_text": "Alice lives in Berlin"}),
    ]
    return nodes, edges


def collect(source):
    async def drain():
        return [record async for record in source.records()]

    return asyncio.run(drain())


class TestMemoryObject:
    def test_appendix_a_validates_and_unknown_fields_are_preserved(self):
        memory = OMPMemory.model_validate({**APPENDIX_A, "x_rank": {"by": "team", "value": 3}})
        assert memory.version == COGXVersion(value=2, comparator="integer")
        assert memory.provenance[0] == COGXContributor(
            id="user:jordan",
            kind="human",
            contribution="instructed",
            at=datetime(2026, 9, 15, 9, 28, tzinfo=timezone.utc),
        )
        assert memory.evidence[1].dangling is True
        assert memory.links == [
            COGXLink(target=APPENDIX_A["identifier"], version=1, rel="supersedes")
        ]
        # OMP 1.3: unknown fields MUST be preserved on export.
        assert memory.extension("x_rank") == {"by": "team", "value": 3}
        assert memory.interchange()["x_rank"] == {"by": "team", "value": 3}
        assert "x_rank" not in memory.runtime_view()

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            OMPMemory.model_validate({"identifier": "m", "media_type": "text/plain", "body": "x"})
        with pytest.raises(ValidationError):
            OMPMemory.model_validate({"identifier": "m", "created_at": WHEN, "body": "x"})

    def test_checksum_is_canonical_and_verifies(self):
        memory = appendix_a()
        # Appendix A's checksum is illustrative: it does not match the canonical form.
        assert memory.verify_checksum() is False
        memory.stamp_checksum()
        assert memory.verify_checksum() is True
        reordered = OMPMemory.model_validate(dict(reversed(list(memory.interchange().items()))))
        assert reordered.compute_checksum() == memory.checksum
        tampered = memory.model_copy(update={"body": memory.body + " Not."})
        assert tampered.verify_checksum() is False
        bare = OMPMemory(identifier="m", media_type="text/plain", body="x", created_at=WHEN)
        assert bare.verify_checksum() is None

    def test_conformance_profiles(self):
        memory = appendix_a().stamp_checksum()
        conformance = memory.conformance()
        assert conformance.profiles == ["minimal", "audit"]
        assert conformance.problems == []
        assert any("dangling" in note for note in conformance.notes)

        anonymous = memory.model_copy(update={"provenance": []}).stamp_checksum()
        anonymous_conformance = anonymous.conformance()
        assert anonymous_conformance.profiles == ["minimal"]
        assert any("kind human" in problem for problem in anonymous_conformance.problems)
        assert any("author unknown" in note for note in anonymous_conformance.notes)

        bad_type = memory.model_copy(update={"media_type": "markdown"}).stamp_checksum()
        assert bad_type.conformance().profiles == []

        expired = memory.model_copy(
            update={"valid_until": datetime(2000, 1, 1, tzinfo=timezone.utc)}
        ).stamp_checksum()
        assert any("valid_until" in note for note in expired.conformance().notes)

    def test_runtime_view_is_a_projection(self):
        view = appendix_a().stamp_checksum().runtime_view()
        assert {"identifier", "media_type", "body", "title", "tags", "created_at", "path"} <= set(
            view
        )
        assert not {"checksum", "scope", "evidence", "weight", "protocol_version"} & set(view)
        assert view["version"] == 2
        assert view["provenance"][0] == {
            "id": "user:jordan",
            "kind": "human",
            "contribution": "instructed",
        }
        assert all(set(entry) <= {"id", "kind", "contribution"} for entry in view["provenance"])
        assert view["links"] == [
            {"target": APPENDIX_A["identifier"], "rel": "supersedes", "version": 1}
        ]

    def test_markdown_roundtrip_is_exact(self):
        memory = appendix_a().stamp_checksum()
        rendered = memory.to_markdown()
        assert rendered.startswith("---\nidentifier: ")
        assert rendered.endswith("---\n\n" + memory.body)
        assert OMPMemory.from_markdown(rendered).interchange() == memory.interchange()
        multiline = memory.model_copy(update={"body": "line one\n\n- a list\n"})
        assert OMPMemory.from_markdown(multiline.to_markdown()).body == multiline.body

    def test_markdown_without_front_matter_is_the_smallest_memory(self):
        """Appendix C: a markdown file plus what its carrier knows conforms."""
        memory = OMPMemory.from_markdown(
            "The user prefers concise answers.\n", identifier="MEMORY.md", created_at=WHEN
        )
        assert memory.identifier == "MEMORY.md"
        assert memory.media_type == "text/markdown"
        assert memory.created_at == WHEN
        assert memory.body == "The user prefers concise answers.\n"
        conformance = memory.conformance()
        assert conformance.profiles == ["minimal"]
        assert any("author unknown" in note for note in conformance.notes)

        with_front_matter = OMPMemory.from_markdown(
            "---\nidentifier: notes.md\nmedia_type: text/markdown\n"
            "created_at: 2026-10-14\nprotocol_version: 0.6\nversion: 2\n---\n\nBody.",
            identifier="ignored.md",
            created_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        assert with_front_matter.identifier == "notes.md"  # the file's own metadata wins
        assert with_front_matter.created_at == WHEN  # a YAML date becomes a UTC datetime
        assert with_front_matter.protocol_version == "0.6"  # a YAML float becomes the string
        assert with_front_matter.version == COGXVersion(value=2, comparator="integer")
        assert with_front_matter.body == "Body."
        with pytest.raises(ValueError):
            OMPMemory.from_markdown("---\n- not a mapping\n---\n\nBody.", identifier="x")


class TestRecordMapping:
    def test_foreign_memory_becomes_a_cogx_memory(self):
        memory = OMPMemory.model_validate(
            {**APPENDIX_A, "scope": ["team:platform", "user:jordan"], "x_rank": 3}
        )
        record = record_from_memory(memory, external_system="store:6f1c")
        assert isinstance(record, COGXMemory)
        assert record.external_system == "store:6f1c"
        assert record.external_id == memory.identifier
        assert record.content == memory.body
        assert record.media_type == "text/markdown"
        assert record.title == memory.title and record.path == memory.path
        assert record.tags == ["preference", "team-process"] and record.categories == []
        assert record.scope_tags == ["team:platform"]
        assert record.scope == COGXScope(user_id="jordan")
        assert record.version == COGXVersion(value=2)
        assert record.provenance == memory.provenance
        assert record.evidence == memory.evidence
        assert record.links == memory.links
        assert record.weight == 1.7
        assert record.checksum == memory.checksum
        assert record.valid_from == memory.valid_from
        assert record.extensions == {"x_rank": 3}  # preserved for the next export

    @pytest.mark.parametrize(
        "record",
        [
            COGXEntity(
                external_id="e1",
                name="Alice",
                entity_type="Person",
                description="A person",
                aliases=["Ali"],
                attributes={"age": 30},
                semantic_kind="fact",
                created_at=WHEN,
                version=3,
                accountable="sam",
                confidence=0.9,
            ),
            COGXFact(
                external_id="f1",
                subject_ref="e1",
                predicate="lives_in",
                object_ref="e2",
                fact_text="Alice lives in Berlin",
                subject=["Alice", "Berlin"],
                valid_at=WHEN,
                created_at=WHEN,
                evidence=[COGXEvidence(ref="c1", relation="inferred", confidence=0.8)],
            ),
            COGXDocument(
                external_id="d1",
                content="# Notes\n\nSome text.",
                mime_type="text/markdown",
                title="Notes",
                created_at=WHEN,
                scope=COGXScope(user_id="u1", run_id="r1"),
                scope_tags=["team:x"],
            ),
            COGXEpisode(
                external_id="ep1",
                title="Standup",
                turns=[
                    COGXTurn(role="user", content="hi", occurred_at=WHEN),
                    COGXTurn(role="assistant", content="hello"),
                ],
                created_at=WHEN,
            ),
            COGXMemory(
                external_id="m1",
                content="Likes tea",
                categories=["preference"],
                tags=["drinks"],
                created_at=WHEN,
                invalidated_by="m2",
                metadata={"mem0_metadata": {"k": "v"}},
            ),
            COGXMemoryBlock(
                external_id="b1", label="persona", value="Helpful.", limit=200, created_at=WHEN
            ),
        ],
        ids=["entity", "fact", "document", "episode", "memory", "memory_block"],
    )
    def test_cognee_records_roundtrip_losslessly(self, record):
        memory = memory_from_record(record, exported_at=WHEN)
        assert memory.extension("cognee")["kind"] == record.kind
        assert memory.identifier == record.external_id
        assert memory.verify_checksum() is True
        assert memory.conformance().minimal
        back = record_from_memory(memory, external_system=record.external_system)
        assert type(back) is type(record)
        exclude = {"checksum", "protocol_version"}
        assert back.model_dump(exclude=exclude) == record.model_dump(exclude=exclude)

    def test_scope_and_tags_projection(self):
        record = COGXMemory(
            external_id="m",
            content="x",
            categories=["preference"],
            semantic_kind="decision",
            scope=COGXScope(user_id="u1", run_id="r1"),
            scope_tags=["team:x"],
            created_at=WHEN,
        )
        memory = memory_from_record(record)
        assert memory.scope == ["team:x", "user:u1", "run:r1"]
        assert memory.tags == ["decision", "preference"]  # where a system records the kind (1.3)
        back = record_from_memory(memory)
        assert back.scope == record.scope and back.scope_tags == ["team:x"]
        assert back.tags == [] and back.categories == ["preference"]
        assert back.semantic_kind == "decision"

    def test_invalidated_by_becomes_a_superseded_by_link(self):
        memory = memory_from_record(COGXMemory(external_id="m1", content="x", invalidated_by="m2"))
        assert memory.links == [COGXLink(target="m2", rel="superseded_by")]
        back = record_from_memory(memory)
        assert back.invalidated_by == "m2" and back.links == []

    def test_missing_created_at_falls_back_to_the_export_time(self):
        memory = memory_from_record(COGXMemory(external_id="m1", content="x"), exported_at=WHEN)
        assert memory.created_at == WHEN
        assert memory_from_record(COGXMemory(external_id="m1", content="x")).created_at.tzinfo

    def test_fact_without_text_renders_its_triple(self):
        named = COGXFact(
            external_id="f",
            subject_ref="e1",
            predicate="lives_in",
            object_ref="e2",
            subject=["Alice", "Berlin"],
        )
        assert memory_from_record(named).body == "Alice lives_in Berlin"
        bare = COGXFact(external_id="f", subject_ref="e1", predicate="lives_in", object_ref="e2")
        assert memory_from_record(bare).body == "e1 lives_in e2"
        assert memory_from_record(bare).media_type == "text/plain"

    def test_mark_dangling_keeps_and_marks(self):
        memory = appendix_a()
        mark_dangling(memory, {APPENDIX_A["identifier"]})
        assert memory.links[0].dangling is False  # resolves within the export
        assert memory.evidence[0].dangling is True  # a transcript turn lives outside the store
        assert memory.evidence[1].dangling is True  # the exporter's own mark is kept
        assert len(memory.links) == 1 and len(memory.evidence) == 2  # nothing dropped


class TestExportAndImport:
    def test_omp_jsonl_export_and_read_back(self, tmp_path):
        nodes, edges = graph()
        destination = tmp_path / "ds_omp"
        result = _write_omp(
            nodes, edges, destination, "ds", NO_FACTS, origin="cognee:ds", layout="jsonl"
        )
        assert result == {"memories": 7, "profiles": ["minimal"], "layout": "jsonl"}
        assert (destination / OMP_MEMORIES_FILE).is_file()

        manifest = read_omp_manifest(destination)
        assert manifest.protocol_version == "0.6"
        assert manifest.origin == "cognee:ds"
        assert manifest.profiles == ["minimal"]
        assert manifest.selection == "snapshot" and manifest.intent == "copy"
        assert manifest.media_types == ["text/plain"]
        assert manifest.model_extra["memories"] == 7
        assert "coverage" in manifest.model_extra["cognee"]

        memories = {memory.identifier: memory for memory in iter_omp_memories(destination)}
        assert DOC not in memories  # structure, not a memory
        assert all(memory.verify_checksum() for memory in memories.values())
        assert all(memory.conformance().minimal for memory in memories.values())
        assert memories[ALICE].title == "Alice"
        assert memories[ALICE].version == COGXVersion(value=1)
        assert memories[ALICE].extension("cognee")["kind"] == "entity"
        assert memories[CHUNK].body == TEXT
        assert memories[CHUNK].links == [
            COGXLink(target=ALICE, rel="mentions"),
            COGXLink(target=BERLIN, rel="mentions"),
        ]
        fact = memories[f"{ALICE}:lives_in:{BERLIN}"]
        assert fact.body == "Alice lives in Berlin"
        assert fact.links == [
            COGXLink(target=ALICE, rel="subject"),
            COGXLink(target=BERLIN, rel="object"),
        ]

    def test_omp_markdown_export(self, tmp_path):
        nodes, edges = graph()
        destination = tmp_path / "ds_md"
        result = _write_omp(
            nodes, edges, destination, "ds", NO_FACTS, origin="cognee:ds", layout="markdown"
        )
        assert result["layout"] == "markdown" and result["memories"] == 7
        files = sorted(path.name for path in destination.glob("*.md"))
        assert OMP_INDEX_FILE in files and f"{ALICE}.md" in files
        assert len(files) == 8  # seven memories plus the root index
        index = OMPMemory.from_markdown((destination / OMP_INDEX_FILE).read_text())
        assert index.stub is True and index.tags == ["index"]
        assert f"[Alice]({ALICE}.md)" in index.body
        memories = {memory.identifier: memory for memory in iter_omp_memories(destination)}
        assert len(memories) == 8
        assert memories[ALICE].links == [] and memories[ALICE].path == f"{ALICE}.md"
        assert memories[CHUNK].body == TEXT

    def test_markdown_export_keeps_files_inside_the_destination(self, tmp_path):
        destination = tmp_path / "out"
        memories = [
            OMPMemory(
                identifier="evil",
                media_type="text/plain",
                body="x",
                created_at=WHEN,
                path="../escape.md",
            ),
            OMPMemory(
                identifier="abs",
                media_type="text/plain",
                body="x",
                created_at=WHEN,
                path="/etc/x.md",
            ),
            OMPMemory(
                identifier="ok", media_type="text/plain", body="x", created_at=WHEN, path="team/ok"
            ),
            OMPMemory(
                identifier="dup", media_type="text/plain", body="y", created_at=WHEN, path="team/ok"
            ),
        ]
        write_omp_export(memories, destination, layout="markdown")
        assert not (tmp_path / "escape.md").exists()
        assert (destination / "evil.md").is_file() and (destination / "abs.md").is_file()
        assert (destination / "team" / "ok.md").is_file() and (
            destination / "team" / "ok-1.md"
        ).is_file()
        with pytest.raises(ValueError, match="layout"):
            write_omp_export([], destination, layout="xml")

    def test_omp_source_imports_a_cognee_export_with_a_receipt(self, tmp_path):
        nodes, edges = graph()
        destination = tmp_path / "ds_omp"
        _write_omp(nodes, edges, destination, "ds", NO_FACTS, origin="cognee:ds", layout="jsonl")

        source = OMPSource(destination, mode="preserve")
        source.receipt = ImportReceipt()
        records = collect(source)
        assert (
            sorted(record.kind for record in records)
            == ["document", "entity", "entity"] + ["fact"] * 4
        )
        assert {record.external_system for record in records} == {"cognee:ds"}
        assert source.receipt.integrity == {"verified": 7, "unverified": 0, "failed": []}
        collect(source)  # a replay verifies nothing twice
        assert source.receipt.integrity["verified"] == 7
        chunk = next(record for record in records if record.kind == "document")
        assert all(link.dangling is False for link in chunk.links)
        part_of = next(
            record
            for record in records
            if record.kind == "fact" and record.predicate == "is_part_of"
        )
        assert [link.dangling for link in part_of.links] == [False, True]  # DOC was not exported

        translation = translate_records(records, "preserve", receipt=source.receipt)
        edges_out = [edge for batch in translation.graph_batches for edge in batch["edges"]]
        assert [edge[2] for edge in edges_out] == ["lives_in"]
        receipt = translation.receipt
        # The chunk is a memory (a data item), not a graph node: the facts that
        # point at it are omitted, kept on the receipt with their dangling refs.
        assert translation.skipped_facts == 3
        assert receipt.counts == {"accepted": 4, "transformed": 0, "omitted": 3, "rejected": 0}
        assert {tuple(entry["refs"]) for entry in receipt.dangling} == {(CHUNK,), (CHUNK, DOC)}
        # The receiving system assigned its own ids: the mapping is kept (3.5.2).
        assert receipt.identifier_map[ALICE] == str(Entity.id_for("Alice"))

    def test_omp_source_reads_a_markdown_folder(self, tmp_path):
        (tmp_path / "MEMORY.md").write_text("The user prefers concise answers.\n")
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "pref.md").write_text(
            "---\nidentifier: pref-1\ntags: [preference]\ncreated_at: 2026-10-14T00:00:00Z\n"
            "---\n\nStatus updates are due Fridays.\n"
        )
        source = OMPSource(tmp_path)
        assert source.manifest is None
        records = {record.external_id: record for record in collect(source)}
        assert set(records) == {"MEMORY.md", "pref-1"}
        assert all(isinstance(record, COGXMemory) for record in records.values())
        assert all(record.external_system == "omp" for record in records.values())
        assert records["MEMORY.md"].media_type == "text/markdown"
        assert records["MEMORY.md"].created_at is not None
        assert records["pref-1"].path == "notes/pref.md"
        assert records["pref-1"].tags == ["preference"]
        assert records["pref-1"].content == "Status updates are due Fridays.\n"

    def test_omp_source_reads_a_json_file_and_reports_integrity(self, tmp_path):
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps({"memories": [APPENDIX_A]}))
        source = OMPSource(export_file)
        source.receipt = ImportReceipt()
        (record,) = collect(source)
        assert record.external_id == APPENDIX_A["identifier"]
        assert record.links[0].dangling is False  # the target is the export's own memory
        assert [entry.dangling for entry in record.evidence] == [True, True]
        # Appendix A's illustrative checksum does not match: reported, not rejected.
        assert source.receipt.integrity["failed"] == [APPENDIX_A["identifier"]]
        assert source.receipt.integrity["failed_count"] == 1
        with pytest.raises(FileNotFoundError):
            collect(OMPSource(tmp_path / "missing.jsonl"))


class TestImportReceipt:
    def test_counts_entries_and_mappings(self):
        receipt = ImportReceipt()
        receipt.record("accepted", "a", kind="memory", local_id="local-a")
        receipt.record("transformed", "b", kind="entity", reason="merged", local_id="local-b")
        receipt.record("omitted", "c", kind="fact", reason="dangling reference", refs=["x"])
        assert receipt.counts == {"accepted": 1, "transformed": 1, "omitted": 1, "rejected": 0}
        assert receipt.identifier_map == {"a": "local-a", "b": "local-b"}
        assert [entry["identifier"] for entry in receipt.entries] == ["b", "c"]
        assert receipt.dangling == [
            {
                "status": "omitted",
                "identifier": "c",
                "kind": "fact",
                "reason": "dangling reference",
                "refs": ["x"],
            }
        ]
        receipt.integrity_check("a", True)
        receipt.integrity_check("b", None)
        summary = receipt.summary()
        assert summary["integrity"] == {"verified": 1, "unverified": 1, "failed": []}
        assert json.dumps(summary)  # JSON-safe
        with pytest.raises(ValueError):
            receipt.record("lost", "d")

    def test_entries_are_bounded_and_the_overflow_counted(self):
        receipt = ImportReceipt()
        for index in range(MAX_RECEIPT_ENTRIES + 5):
            receipt.record("omitted", f"m-{index}", reason="dangling reference", refs=["x"])
        assert len(receipt.entries) == MAX_RECEIPT_ENTRIES
        assert receipt.entries_dropped == 5
        assert receipt.counts["omitted"] == MAX_RECEIPT_ENTRIES + 5


class TestExportDatasetOMP:
    def test_export_dataset_writes_an_omp_export(self, tmp_path, monkeypatch):
        """The public export path: ``format="omp"`` and ``"omp-markdown"``
        reach the OMP writer with the dataset as the store origin, and the
        default destination is ``<dataset>_omp``."""
        from contextlib import asynccontextmanager
        from types import SimpleNamespace
        from uuid import uuid4

        import cognee.context_global_variables as global_variables
        import cognee.infrastructure.databases.graph as graph_module
        import cognee.modules.data.methods as data_methods
        import cognee.modules.migration.export as export_module
        from cognee.modules.migration.export import export_dataset

        nodes, edges = graph()
        dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4(), name="ds")

        @asynccontextmanager
        async def fake_context(*args, **kwargs):
            yield

        async def fake_get_graph_engine():
            async def get_graph_data():
                return nodes, edges

            return SimpleNamespace(get_graph_data=get_graph_data)

        async def fake_get_authorized_existing_datasets(datasets, permission, user):
            return [dataset]

        async def fake_load_export_facts(dataset_id, nodes, edges):
            return NO_FACTS

        async def fake_stored_migration_revision(dataset_id):
            return "some_slug"

        monkeypatch.setattr(global_variables, "set_database_global_context_variables", fake_context)
        monkeypatch.setattr(graph_module, "get_graph_engine", fake_get_graph_engine)
        monkeypatch.setattr(
            data_methods, "get_authorized_existing_datasets", fake_get_authorized_existing_datasets
        )
        monkeypatch.setattr(export_module, "_load_export_facts", fake_load_export_facts)
        monkeypatch.setattr(
            export_module, "_stored_migration_revision", fake_stored_migration_revision
        )
        monkeypatch.chdir(tmp_path)
        user = SimpleNamespace(id=uuid4())

        result = asyncio.run(export_dataset("ds", format="omp", user=user))
        assert result.format == "omp" and result.destination == "ds_omp"
        manifest = read_omp_manifest(tmp_path / "ds_omp")
        assert manifest.origin == f"cognee:{dataset.id}"
        assert manifest.model_extra["cognee"]["migration_revision"] == "some_slug"
        assert manifest.model_extra["memories"] == 7
        assert len(list(iter_omp_memories(tmp_path / "ds_omp"))) == 7

        markdown = asyncio.run(
            export_dataset("ds", format="omp-markdown", destination=tmp_path / "md", user=user)
        )
        assert markdown.format == "omp-markdown"
        assert (tmp_path / "md" / OMP_INDEX_FILE).is_file()
        assert read_omp_manifest(tmp_path / "md").model_extra["layout"] == "markdown"

        with pytest.raises(ValueError, match="Unknown export format"):
            asyncio.run(export_dataset("ds", format="omp-yaml", user=user))
