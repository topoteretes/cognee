"""Round-trip tests for the real export mapping: _write_cogx -> read_archive -> translate_records.

These exercise the cognee -> COGX -> cognee path on synthetic (nodes, edges)
shaped exactly like ``get_graph_data()`` output — Entity, DocumentChunk,
EntityType, and TextSummary nodes with contains/is_a edges — without touching
a database. All tests are pure: no databases, no LLM calls, no network.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from cognee.modules.migration import loader
from cognee.modules.migration.archive import pack_archive, unpack_archive
from cognee.modules.migration.cogx import COGX_VERSION, read_archive, read_manifest
from cognee.modules.migration.export import _write_cogx
from cognee.modules.migration.loader import translate_records
from cognee.modules.migration.sources.cogx_archive import COGXArchiveSource

ALICE_ID = "11111111-1111-4111-8111-111111111111"
BERLIN_ID = "22222222-2222-4222-8222-222222222222"
PERSON_TYPE_ID = "33333333-3333-4333-8333-333333333333"
CHUNK_ID = "44444444-4444-4444-8444-444444444444"
SUMMARY_ID = "55555555-5555-4555-8555-555555555555"


def sample_graph():
    """A small cognee graph in get_graph_data() shape.

    Nodes are ``(node_id, properties)`` tuples; edges are ``(source_id,
    target_id, relationship_name, properties)`` tuples.
    """
    nodes = [
        (ALICE_ID, {"type": "Entity", "name": "Alice", "description": "A person"}),
        (BERLIN_ID, {"type": "Entity", "name": "Berlin", "description": "A city"}),
        (PERSON_TYPE_ID, {"type": "EntityType", "name": "Person", "description": "Person"}),
        (
            CHUNK_ID,
            {"type": "DocumentChunk", "text": "Alice lives in Berlin.", "chunk_index": 0},
        ),
        (SUMMARY_ID, {"type": "TextSummary", "text": "Alice is a Berlin resident."}),
    ]
    edges = [
        (CHUNK_ID, ALICE_ID, "contains", {}),
        (CHUNK_ID, BERLIN_ID, "contains", {}),
        (ALICE_ID, PERSON_TYPE_ID, "is_a", {}),
        (ALICE_ID, BERLIN_ID, "lives_in", {"edge_text": "Alice lives in Berlin"}),
        (SUMMARY_ID, CHUNK_ID, "made_from", {}),
    ]
    return nodes, edges


def _is_uuid_string(value):
    try:
        UUID(str(value))
        return True
    except (ValueError, TypeError):
        return False


def _roundtrip(tmp_path, mode="preserve"):
    nodes, edges = sample_graph()
    destination = tmp_path / "archive_cogx"
    _write_cogx(nodes, edges, destination, "main_dataset")
    return translate_records(read_archive(destination), mode)


def _all_nodes(result):
    return [node for batch in result.graph_batches for node in batch["nodes"]]


def _all_edges(result):
    return [edge for batch in result.graph_batches for edge in batch["edges"]]


class TestPreserveRoundTrip:
    def test_no_uuid_named_entities(self, tmp_path):
        """C1 regression: edge endpoints must never become UUID-named entities."""
        result = _roundtrip(tmp_path)
        assert result.skipped_facts == 0
        names = [getattr(node, "name", None) for node in _all_nodes(result)]
        uuid_names = [name for name in names if name and _is_uuid_string(name)]
        assert uuid_names == []
        # The real entity vocabulary survived.
        assert {"Alice", "Berlin"} <= set(names)

    def test_all_edges_restored_onto_original_topology(self, tmp_path):
        result = _roundtrip(tmp_path)
        edges = _all_edges(result)
        assert len(edges) == 5
        assert {edge[2] for edge in edges} == {"contains", "is_a", "lives_in", "made_from"}

        lives_in = next(edge for edge in edges if edge[2] == "lives_in")
        assert lives_in[3]["edge_text"] == "Alice lives in Berlin"

        # Chunk-referencing facts attach to the restored raw chunk node, not a stub.
        contains = [edge for edge in edges if edge[2] == "contains"]
        assert all(str(edge[0]) == CHUNK_ID for edge in contains)
        is_a = next(edge for edge in edges if edge[2] == "is_a")
        assert str(is_a[1]) == PERSON_TYPE_ID

    def test_raw_nodes_yielded_and_restored(self, tmp_path):
        """H1: nodes.jsonl is read back and rehydrated into the graph batches."""
        nodes, edges = sample_graph()
        destination = tmp_path / "archive_cogx"
        _write_cogx(nodes, edges, destination, "main_dataset")

        records = list(read_archive(destination))
        raw_nodes = [record for record in records if record.kind == "raw_node"]
        # EntityType, TextSummary, and the dual-written DocumentChunk survive raw.
        assert {record.properties["id"] for record in raw_nodes} == {
            PERSON_TYPE_ID,
            CHUNK_ID,
            SUMMARY_ID,
        }
        # The chunk is also a document, so re-derive/cross-provider keeps its content.
        documents = [record for record in records if record.kind == "document"]
        assert [document.external_id for document in documents] == [CHUNK_ID]
        assert documents[0].content == "Alice lives in Berlin."

        result = translate_records(records, "preserve")
        node_ids = {str(node.id) for node in _all_nodes(result)}
        assert {PERSON_TYPE_ID, CHUNK_ID, SUMMARY_ID} <= node_ids

    def test_re_derive_roundtrip_keeps_chunk_content(self, tmp_path):
        result = _roundtrip(tmp_path, mode="re-derive")
        assert result.cognify_data_items is True
        assert result.graph_batches == []
        contents = [item.data for item in result.data_items]
        assert any("Alice lives in Berlin." in content for content in contents)


class TestReExport:
    def test_reexport_same_destination_does_not_duplicate(self, tmp_path):
        """H2: re-exporting into an existing destination starts from a clean slate."""
        nodes, edges = sample_graph()
        destination = tmp_path / "archive_cogx"
        _write_cogx(nodes, edges, destination, "main_dataset")
        first_files = sorted(path.name for path in destination.iterdir())
        first_counts = read_manifest(destination).counts

        _write_cogx(nodes, edges, destination, "main_dataset")

        assert sorted(path.name for path in destination.iterdir()) == first_files
        manifest = read_manifest(destination)
        assert manifest.counts == first_counts
        # JSONL line counts match the manifest: nothing was appended.
        for file_name, kind in (
            ("entities.jsonl", "entity"),
            ("facts.jsonl", "fact"),
            ("nodes.jsonl", "raw_node"),
        ):
            lines = (destination / file_name).read_text().splitlines()
            assert len(lines) == manifest.counts[kind]

        result = translate_records(read_archive(destination), "preserve")
        assert result.skipped_facts == 0
        assert len(_all_edges(result)) == 5


class TestBatchingBounds:
    def test_large_roundtrip_is_split_into_bounded_batches(self, tmp_path, monkeypatch):
        """H5: a >batch-size export yields multiple self-contained graph batches."""
        monkeypatch.setattr(loader, "BATCH_NODE_TARGET", 4)
        nodes = [
            (str(uuid4()), {"type": "Entity", "name": f"Entity {i}", "description": f"d{i}"})
            for i in range(12)
        ]
        edges = [(nodes[i][0], nodes[i + 1][0], "knows", {}) for i in range(11)]
        edges.append((nodes[0][0], nodes[11][0], "mentions", {}))

        destination = tmp_path / "archive_cogx"
        _write_cogx(nodes, edges, destination, "main_dataset")
        result = translate_records(read_archive(destination), "preserve")

        assert result.skipped_facts == 0
        assert len(result.graph_batches) == 3
        for batch in result.graph_batches:
            # Bounded: the target plus at most one duplicated endpoint per edge.
            assert len(batch["nodes"]) <= 4 + len(batch["edges"])
            # Self-contained: every edge's endpoints live in its own batch.
            batch_node_ids = {node.id for node in batch["nodes"]}
            for source_id, target_id, _, _ in batch["edges"]:
                assert source_id in batch_node_ids
                assert target_id in batch_node_ids

        # Every edge restored exactly once; duplicates share deterministic ids.
        assert len(_all_edges(result)) == 12
        assert len({str(node.id) for node in _all_nodes(result)}) == 12


class TestManifest:
    def test_embedding_model_and_version_stamped(self, tmp_path):
        nodes, edges = sample_graph()
        destination = tmp_path / "archive_cogx"
        _write_cogx(
            nodes, edges, destination, "main_dataset", embedding_model="text-embedding-3-large"
        )
        manifest = read_manifest(destination)
        assert manifest.cogx_version == COGX_VERSION
        assert manifest.embedding_model == "text-embedding-3-large"
        assert manifest.source_system == "cognee"
        assert manifest.counts == {"entity": 2, "document": 1, "raw_node": 3, "fact": 5}

    def test_embedding_model_defaults_to_none(self, tmp_path):
        nodes, edges = sample_graph()
        destination = tmp_path / "archive_cogx"
        _write_cogx(nodes, edges, destination, "main_dataset")
        assert read_manifest(destination).embedding_model is None

    def _tamper_version(self, destination, version):
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cogx_version"] = version
        manifest_path.write_text(json.dumps(manifest))

    def test_future_major_version_raises(self, tmp_path):
        """L7: archives from a newer major COGX version are rejected up front."""
        nodes, edges = sample_graph()
        destination = tmp_path / "archive_cogx"
        _write_cogx(nodes, edges, destination, "main_dataset")
        self._tamper_version(destination, "99.0")
        with pytest.raises(ValueError, match="99.0"):
            read_manifest(destination)

    def test_future_minor_version_accepted(self, tmp_path):
        nodes, edges = sample_graph()
        destination = tmp_path / "archive_cogx"
        _write_cogx(nodes, edges, destination, "main_dataset")
        self._tamper_version(destination, "0.99")
        assert read_manifest(destination).cogx_version == "0.99"


class TestTimestampFidelity:
    def test_micro_and_nano_epochs_survive_export(self, tmp_path):
        """L2: finer-than-millisecond epochs are scaled down, never crash the export."""
        expected = datetime.fromtimestamp(1709294400, tz=timezone.utc)
        nodes = [
            (
                ALICE_ID,
                {
                    "type": "Entity",
                    "name": "Alice",
                    "description": "A person",
                    "created_at": 1709294400000000,  # microseconds
                    "updated_at": 1709294400000000000,  # nanoseconds
                },
            )
        ]
        destination = tmp_path / "archive_cogx"
        _write_cogx(nodes, [], destination, "main_dataset")
        entity = next(record for record in read_archive(destination) if record.kind == "entity")
        assert entity.created_at == expected
        assert entity.updated_at == expected


# A hand-written v0.1 archive: the on-disk wire format frozen exactly as it
# ships today. If a future writer change alters the format, re-importing this
# fixture must still rebuild the graph — that is the backwards-compatibility
# contract for archives produced by older cognee versions / existing backups.
FROZEN_MANIFEST = json.dumps(
    {
        "cogx_version": "0.1",
        "source_system": "cognee",
        "counts": {"entity": 2, "fact": 1, "raw_node": 1},
    }
)
FROZEN_ENTITIES = (
    '{"kind": "entity", "external_system": "cognee", "external_id": "ent-alice", "name": "Alice", "description": "A person"}\n'
    '{"kind": "entity", "external_system": "cognee", "external_id": "ent-berlin", "name": "Berlin", "description": "A city"}\n'
)
FROZEN_FACTS = (
    '{"kind": "fact", "external_system": "cognee", "external_id": "fact-1", '
    '"subject_ref": "ent-alice", "predicate": "lives_in", "object_ref": "ent-berlin", '
    '"fact_text": "Alice lives in Berlin"}\n'
)
FROZEN_NODES = (
    '{"id": "' + PERSON_TYPE_ID + '", "type": "EntityType", "name": "Person", "description": "Person"}\n'
)


def _collect_records(source):
    async def drain():
        return [record async for record in source.records()]

    return asyncio.run(drain())


class TestArchiveTransportRoundTrip:
    """End-to-end export -> pack -> unpack -> re-import through the public
    ``COGXArchiveSource`` — the path ``cognee.push()`` and restore actually use.

    Unlike the mapping tests above (which read the archive directory directly),
    this drives the real tar transport and the public re-import entry point, so
    it is the regression anchor the reviewer asked for: a format/transport
    change that breaks restore fails here.
    """

    def test_export_pack_unpack_reimport_preserves_graph(self, tmp_path):
        nodes, edges = sample_graph()
        archive_dir = tmp_path / "export_cogx"
        _write_cogx(nodes, edges, archive_dir, "main_dataset")

        # Transport: tar the archive, then unpack it into a fresh location, as a
        # restore on another instance (or after a backup) would.
        tar_path = pack_archive(archive_dir, tmp_path / "export.cogx.tar.gz")
        with open(tar_path, "rb") as handle:
            restore_root = unpack_archive(handle, tmp_path / "restored")

        # Re-import through the public source (defaults to preserve mode).
        source = COGXArchiveSource(restore_root)
        assert source.source_system == "cognee"
        result = translate_records(_collect_records(source), source.mode)

        # The full topology survives the round-trip with nothing skipped or
        # fabricated as a UUID-named entity.
        assert result.skipped_facts == 0
        edges_out = _all_edges(result)
        assert len(edges_out) == 5
        assert {edge[2] for edge in edges_out} == {"contains", "is_a", "lives_in", "made_from"}
        names = [getattr(node, "name", None) for node in _all_nodes(result)]
        assert {"Alice", "Berlin"} <= set(names)
        assert not any(name and _is_uuid_string(name) for name in names)
        node_ids = {str(node.id) for node in _all_nodes(result)}
        assert {PERSON_TYPE_ID, CHUNK_ID, SUMMARY_ID} <= node_ids

    def test_frozen_archive_format_still_imports(self, tmp_path):
        archive_dir = tmp_path / "frozen_cogx"
        archive_dir.mkdir()
        (archive_dir / "manifest.json").write_text(FROZEN_MANIFEST, encoding="utf-8")
        (archive_dir / "entities.jsonl").write_text(FROZEN_ENTITIES, encoding="utf-8")
        (archive_dir / "facts.jsonl").write_text(FROZEN_FACTS, encoding="utf-8")
        (archive_dir / "nodes.jsonl").write_text(FROZEN_NODES, encoding="utf-8")

        source = COGXArchiveSource(archive_dir)
        records = _collect_records(source)
        assert sorted(record.kind for record in records) == ["entity", "entity", "fact", "raw_node"]

        result = translate_records(records, "preserve")
        assert result.skipped_facts == 0
        edges_out = _all_edges(result)
        assert [edge[2] for edge in edges_out] == ["lives_in"]
        lives_in = edges_out[0]
        assert lives_in[3]["edge_text"] == "Alice lives in Berlin"
        node_ids = {str(node.id) for node in _all_nodes(result)}
        assert PERSON_TYPE_ID in node_ids  # raw node rehydrated


class TestExportDatasetSelfLoops:
    def test_self_loops_filtered_before_emitters_and_counts(self, tmp_path, monkeypatch):
        """M1: synthesized SELF self-loops never reach an emitter or num_edges."""
        import cognee.context_global_variables as global_variables
        import cognee.infrastructure.databases.graph as graph_module
        import cognee.modules.data.methods as data_methods

        from cognee.modules.migration.export import export_dataset

        nodes, edges = sample_graph()
        # Ladybug's get_graph_data() fabricates a SELF self-loop per node when
        # the graph has no real edges; simulate a pair of them.
        self_loops = [
            (ALICE_ID, ALICE_ID, "SELF", {}),
            (BERLIN_ID, BERLIN_ID, "SELF", {}),
        ]
        dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4(), name="main_dataset")

        @asynccontextmanager
        async def fake_context(*args, **kwargs):
            yield

        async def fake_get_graph_data():
            return nodes, edges + self_loops

        async def fake_get_graph_engine():
            return SimpleNamespace(get_graph_data=fake_get_graph_data)

        async def fake_get_authorized_existing_datasets(datasets, permission, user):
            return [dataset]

        monkeypatch.setattr(global_variables, "set_database_global_context_variables", fake_context)
        monkeypatch.setattr(graph_module, "get_graph_engine", fake_get_graph_engine)
        monkeypatch.setattr(
            data_methods,
            "get_authorized_existing_datasets",
            fake_get_authorized_existing_datasets,
        )

        destination = tmp_path / "main_cogx"
        result = asyncio.run(
            export_dataset(
                "main_dataset",
                format="cogx",
                destination=destination,
                user=SimpleNamespace(id=uuid4()),
            )
        )

        assert result.num_nodes == 5
        assert result.num_edges == 5  # the SELF pair is excluded from the count
        facts = [record for record in read_archive(destination) if record.kind == "fact"]
        assert len(facts) == 5
        assert all(fact.predicate != "SELF" for fact in facts)

        # The filtered archive round-trips cleanly with no fabricated names.
        translated = translate_records(read_archive(destination), "preserve")
        assert translated.skipped_facts == 0
        names = [getattr(node, "name", None) for node in _all_nodes(translated)]
        assert not any(name and _is_uuid_string(name) for name in names)
