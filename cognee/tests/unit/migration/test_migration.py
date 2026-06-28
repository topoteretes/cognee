"""Unit tests for the memory-migration module (COGX, sources, loader, formats).

All tests are pure: no databases, no LLM calls.
"""

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from cognee.modules.migration.cogx import (
    COGX_VERSION,
    COGXArchiveWriter,
    COGXDocument,
    COGXEntity,
    COGXEpisode,
    COGXFact,
    COGXMemory,
    COGXMemoryBlock,
    COGXRawNode,
    COGXTurn,
    parse_timestamp,
    read_archive,
    read_manifest,
)
from cognee.modules.migration.formats import write_cypher, write_graphml, write_json
from cognee.modules.migration import loader
from cognee.modules.migration.loader import record_data_id, translate_records
from cognee.modules.migration.sources import (
    COGXArchiveSource,
    GraphitiSource,
    LangMemSource,
    LettaSource,
    Mem0Source,
)


def collect(source):
    async def _collect():
        return [record async for record in source.records()]

    return asyncio.run(_collect())


class TestParseTimestamp:
    def test_iso_string(self):
        parsed = parse_timestamp("2024-03-01T12:00:00Z")
        assert parsed == datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)

    def test_epoch_seconds_and_milliseconds(self):
        seconds = parse_timestamp(1709294400)
        milliseconds = parse_timestamp(1709294400000)
        assert seconds == milliseconds

    def test_epoch_micro_and_nanoseconds(self):
        seconds = parse_timestamp(1709294400)
        assert parse_timestamp(1709294400000000) == seconds
        assert parse_timestamp(1709294400000000000) == seconds

    def test_invalid_values(self):
        assert parse_timestamp(None) is None
        assert parse_timestamp("not-a-date") is None
        # Out-of-range epochs return None instead of raising.
        assert parse_timestamp(float("inf")) is None


class TestCOGXArchive:
    def _sample_records(self):
        return [
            COGXDocument(external_system="test", external_id="d1", content="hello"),
            COGXEpisode(
                external_system="test",
                external_id="e1",
                turns=[COGXTurn(role="user", content="hi")],
            ),
            COGXEntity(external_system="test", external_id="n1", name="Alice"),
            COGXFact(
                external_system="test",
                external_id="f1",
                subject_ref="n1",
                predicate="works_at",
                object_ref="Acme",
                valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            COGXMemory(external_system="test", external_id="m1", content="likes tea"),
            COGXMemoryBlock(external_system="test", external_id="b1", label="persona", value="x"),
        ]

    def test_roundtrip(self, tmp_path):
        archive_dir = tmp_path / "archive"
        records = self._sample_records()
        with COGXArchiveWriter(archive_dir, source_system="test") as writer:
            for record in records:
                writer.write(record)

        loaded = list(read_archive(archive_dir))
        assert {r.kind for r in loaded} == {r.kind for r in records}
        assert len(loaded) == len(records)

        fact = next(r for r in loaded if r.kind == "fact")
        assert fact.predicate == "works_at"
        assert fact.valid_at == datetime(2024, 1, 1, tzinfo=timezone.utc)

        manifest = read_manifest(archive_dir)
        assert manifest.source_system == "test"
        assert manifest.counts["fact"] == 1

    def test_archive_source_streams_records(self, tmp_path):
        archive_dir = tmp_path / "archive"
        with COGXArchiveWriter(archive_dir, source_system="test") as writer:
            for record in self._sample_records():
                writer.write(record)

        source = COGXArchiveSource(archive_dir)
        assert source.source_system == "test"
        assert source.mode == "preserve"
        assert len(collect(source)) == 6

    def test_raw_nodes_roundtrip(self, tmp_path):
        archive_dir = tmp_path / "archive"
        raw = {"id": "11111111-1111-1111-1111-111111111111", "type": "EntityType", "name": "Person"}
        with COGXArchiveWriter(archive_dir, source_system="test") as writer:
            writer.write(self._sample_records()[2])
            writer.write_raw_node(raw)

        loaded = list(read_archive(archive_dir))
        raw_nodes = [record for record in loaded if record.kind == "raw_node"]
        assert len(raw_nodes) == 1
        assert isinstance(raw_nodes[0], COGXRawNode)
        assert raw_nodes[0].properties == raw
        assert read_manifest(archive_dir).counts["raw_node"] == 1

    def test_writer_starts_from_clean_slate(self, tmp_path):
        archive_dir = tmp_path / "archive"
        for _ in range(2):
            with COGXArchiveWriter(archive_dir, source_system="test") as writer:
                writer.write(self._sample_records()[2])
                writer.write_raw_node({"id": "raw-1", "type": "NodeSet"})

        assert len((archive_dir / "entities.jsonl").read_text().splitlines()) == 1
        assert len((archive_dir / "nodes.jsonl").read_text().splitlines()) == 1
        manifest = read_manifest(archive_dir)
        assert manifest.counts == {"entity": 1, "raw_node": 1}

    def test_future_major_version_rejected(self, tmp_path):
        archive_dir = tmp_path / "archive"
        with COGXArchiveWriter(archive_dir, source_system="test") as writer:
            writer.write(self._sample_records()[2])
        manifest_path = archive_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["cogx_version"] = "99.0"
        manifest_path.write_text(json.dumps(manifest))

        try:
            COGXArchiveSource(archive_dir)
            raise AssertionError("expected ValueError")
        except ValueError as error:
            assert "99.0" in str(error)
            assert COGX_VERSION in str(error)


class TestMem0Source:
    def test_plain_list(self):
        memories = collect(
            Mem0Source(
                [
                    {
                        "id": "abc",
                        "memory": "Alice works at Acme",
                        "user_id": "u1",
                        "categories": ["work"],
                        "created_at": "2024-01-01T00:00:00Z",
                    }
                ]
            )
        )
        assert len(memories) == 1
        assert memories[0].kind == "memory"
        assert memories[0].external_id == "abc"
        assert memories[0].scope.user_id == "u1"
        assert memories[0].categories == ["work"]

    def test_results_wrapper_and_text_key(self):
        memories = collect(Mem0Source({"results": [{"id": "1", "text": "remembers things"}]}))
        assert len(memories) == 1
        assert memories[0].content == "remembers things"

    def test_export_file(self, tmp_path):
        export_file = tmp_path / "mem0.json"
        export_file.write_text(json.dumps([{"id": "1", "memory": "from file"}]))
        memories = collect(Mem0Source(export_file))
        assert memories[0].content == "from file"

    def test_skips_items_without_content(self):
        assert collect(Mem0Source([{"id": "1"}, {"id": "2", "memory": "kept"}])) != []


class TestLangMemSource:
    def test_semantic_memory(self):
        records = collect(
            LangMemSource(
                [
                    {
                        "namespace": ["user-42", "memories"],
                        "key": "mem-001",
                        "value": {
                            "kind": "Memory",
                            "content": {"content": "User prefers dark mode"},
                        },
                        "created_at": "2024-01-01T00:00:00Z",
                    }
                ]
            )
        )
        assert len(records) == 1
        assert records[0].kind == "memory"
        assert records[0].external_id == "mem-001"
        assert records[0].content == "User prefers dark mode"
        assert records[0].scope.user_id == "user-42"
        assert records[0].scope.session_id == "memories"

    def test_items_wrapper(self):
        records = collect(
            LangMemSource(
                {
                    "items": [
                        {
                            "key": "1",
                            "value": {"kind": "Memory", "content": "likes tea"},
                        }
                    ]
                }
            )
        )
        assert records[0].content == "likes tea"

    def test_episodic_episode(self):
        records = collect(
            LangMemSource(
                [
                    {
                        "key": "ep-1",
                        "value": {
                            "kind": "Episodic",
                            "content": {
                                "messages": [
                                    {"role": "user", "content": "hello"},
                                    {"role": "assistant", "content": "hi there"},
                                    {"role": "system", "content": "ignored"},
                                ]
                            },
                        },
                    }
                ]
            )
        )
        assert len(records) == 1
        assert records[0].kind == "episode"
        assert len(records[0].turns) == 2

    def test_procedural_memory_block(self):
        records = collect(
            LangMemSource(
                [
                    {
                        "key": "proc-1",
                        "value": {
                            "kind": "Procedural",
                            "content": "Confirm before delete.",
                        },
                    }
                ]
            )
        )
        assert records[0].kind == "memory_block"
        assert records[0].label == "procedural"
        assert records[0].value == "Confirm before delete."

    def test_entities_and_facts(self):
        records = collect(
            LangMemSource(
                [
                    {
                        "key": "bundle-1",
                        "value": {
                            "kind": "Memory",
                            "content": {"content": "ignored when graph present"},
                            "entities": [{"id": "n1", "name": "Alice", "labels": ["Person"]}],
                            "facts": [
                                {
                                    "id": "f1",
                                    "subject_ref": "n1",
                                    "predicate": "knows",
                                    "object_ref": "Bob",
                                    "fact": "Alice knows Bob",
                                }
                            ],
                        },
                    }
                ]
            )
        )
        kinds = [record.kind for record in records]
        assert kinds.count("entity") == 1
        assert kinds.count("fact") == 1
        assert kinds.count("memory") == 1

    def test_export_file(self, tmp_path):
        fixture = Path(__file__).parent / "fixtures" / "langmem_export.json"
        records = collect(LangMemSource(fixture))
        assert [record.kind for record in records] == ["memory", "episode", "memory_block"]

    def test_skips_items_without_content(self):
        assert collect(LangMemSource([{"key": "empty"}, {"key": "x", "value": {}}])) == []


class TestLettaSource:
    def test_agent_file(self):
        agent_file = {
            "agents": [
                {
                    "name": "assistant",
                    "core_memory": [
                        {"id": "blk1", "label": "persona", "value": "I am helpful", "limit": 2000},
                        {"label": "human", "value": ""},
                    ],
                    "messages": [
                        {"role": "user", "content": "hello", "created_at": "2024-01-01T00:00:00Z"},
                        {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
                        {"role": "system", "content": "ignored"},
                    ],
                    "archival_memory": [{"id": "p1", "text": "archived note"}],
                }
            ]
        }
        records = collect(LettaSource(agent_file))
        kinds = [record.kind for record in records]
        assert kinds.count("memory_block") == 1  # empty block skipped
        assert kinds.count("episode") == 1
        assert kinds.count("document") == 1

        episode = next(r for r in records if r.kind == "episode")
        assert len(episode.turns) == 2  # system message excluded
        assert episode.scope.agent_id == "assistant"

        block = next(r for r in records if r.kind == "memory_block")
        assert block.label == "persona"
        assert block.limit == 2000


class TestZepSource:
    def test_graphiti_export(self):
        export = {
            "episodes": [
                {
                    "uuid": "ep1",
                    "name": "chat",
                    "content": "Alice said she moved to Berlin",
                    "created_at": "2024-02-01T00:00:00Z",
                    "group_id": "g1",
                }
            ],
            "nodes": [
                {
                    "uuid": "n1",
                    "name": "Alice",
                    "labels": ["Entity", "Person"],
                    "summary": "A person",
                },
                {"uuid": "n2", "name": "Berlin"},
            ],
            "edges": [
                {
                    "uuid": "f1",
                    "source_node_uuid": "n1",
                    "target_node_uuid": "n2",
                    "name": "LIVES_IN",
                    "fact": "Alice lives in Berlin",
                    "valid_at": "2024-02-01T00:00:00Z",
                    "invalid_at": None,
                    "episodes": ["ep1"],
                }
            ],
        }
        records = collect(GraphitiSource(export))
        assert [record.kind for record in records] == ["episode", "entity", "entity", "fact"]

        entity = next(r for r in records if r.kind == "entity")
        assert entity.entity_type == "Person"

        fact = records[-1]
        assert fact.subject_ref == "n1"
        assert fact.fact_text == "Alice lives in Berlin"
        assert fact.valid_at is not None
        assert fact.invalid_at is None
        assert fact.provenance == ["ep1"]

    def test_default_mode_is_hybrid(self):
        assert GraphitiSource({"nodes": []}).mode == "hybrid"


class TestTranslateRecords:
    def _records(self):
        return [
            COGXMemory(external_system="mem0", external_id="m1", content="likes tea"),
            COGXEpisode(
                external_system="zep",
                external_id="e1",
                turns=[
                    COGXTurn(
                        role="user",
                        content="hello",
                        occurred_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    )
                ],
            ),
            COGXEntity(
                external_system="zep",
                external_id="n1",
                name="Alice",
                entity_type="Person",
                description="A person",
                aliases=["Alice Smith"],
            ),
            COGXFact(
                external_system="zep",
                external_id="f1",
                subject_ref="n1",
                predicate="lives_in",
                object_ref="Berlin",
                fact_text="Alice lives in Berlin",
                valid_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
        ]

    def test_re_derive_mode(self):
        result = translate_records(self._records(), "re-derive")
        assert result.cognify_data_items is True
        assert result.graph_batches == []
        # memory + episode + entity digest + fact digest
        assert len(result.data_items) == 4
        assert all(item.data_id is not None for item in result.data_items)
        # Episode transcript carries the timestamp.
        episode_item = result.data_items[1]
        assert "2024-01-01" in episode_item.data
        # Fact digest carries validity qualifiers.
        fact_digest = result.data_items[3]
        assert "valid from 2024-02-01" in fact_digest.data

    def test_deterministic_data_ids(self):
        record = COGXMemory(external_system="mem0", external_id="m1", content="x")
        assert record_data_id(record) == record_data_id(record)
        first = translate_records(self._records(), "re-derive").data_items[0]
        second = translate_records(self._records(), "re-derive").data_items[0]
        assert first.data_id == second.data_id

    def test_preserve_mode(self):
        result = translate_records(self._records(), "preserve")
        assert result.cognify_data_items is False
        # Textual records still become (raw) data items.
        assert len(result.data_items) == 2
        assert len(result.graph_batches) == 1

        batch = result.graph_batches[0]
        node_names = {getattr(node, "name", None) for node in batch["nodes"]}
        # Alice, Berlin (created from unresolved ref), and the Person entity type.
        assert {"Alice", "Berlin", "Person"} <= node_names

        assert len(batch["edges"]) == 1
        source_id, target_id, relationship, properties = batch["edges"][0]
        assert relationship == "lives_in"
        assert properties["edge_text"] == "Alice lives in Berlin"
        assert properties["valid_at"].startswith("2024-02-01")
        assert properties["source_system"] == "zep"

        alice = next(node for node in batch["nodes"] if getattr(node, "name", "") == "Alice")
        assert "Also known as: Alice Smith" in alice.description
        assert alice.id == source_id

    def test_entity_merge_by_name(self):
        records = [
            COGXEntity(external_system="a", external_id="x1", name="Alice"),
            COGXEntity(external_system="b", external_id="x2", name="Alice"),
        ]
        batch = translate_records(records, "preserve").graph_batches[0]
        entities = [node for node in batch["nodes"] if type(node).__name__ == "Entity"]
        assert len(entities) == 1

    def test_entity_merge_combines_descriptions(self):
        records = [
            COGXEntity(external_system="a", external_id="x1", name="Alice", description="Founder"),
            COGXEntity(
                external_system="b",
                external_id="x2",
                name="Alice",
                description="Engineer",
                aliases=["Alice Smith"],
                entity_type="Person",
            ),
        ]
        batch = translate_records(records, "preserve").graph_batches[0]
        alice = next(node for node in batch["nodes"] if getattr(node, "name", "") == "Alice")
        assert "Founder" in alice.description
        assert "Engineer" in alice.description
        assert "Alice Smith" in alice.description
        assert alice.is_a is not None and alice.is_a.name == "Person"

    def test_raw_nodes_resolve_uuid_fact_refs(self):
        chunk_id = "0c113fd0-1111-2222-3333-444444444444"
        records = [
            COGXRawNode(properties={"id": chunk_id, "type": "DocumentChunk", "text": "hello"}),
            COGXEntity(
                external_system="cognee",
                external_id="55555555-5555-5555-5555-555555555555",
                name="Alice",
            ),
            COGXFact(
                external_system="cognee",
                external_id="f1",
                subject_ref=chunk_id,
                predicate="contains",
                object_ref="55555555-5555-5555-5555-555555555555",
            ),
        ]
        result = translate_records(records, "preserve")
        assert result.skipped_facts == 0
        batch = result.graph_batches[0]
        # The raw node was rehydrated with its original id, not fabricated.
        assert any(str(node.id) == chunk_id for node in batch["nodes"])
        assert not any(getattr(node, "name", "") == chunk_id for node in batch["nodes"])
        assert len(batch["edges"]) == 1
        assert str(batch["edges"][0][0]) == chunk_id

    def test_unresolved_uuid_fact_refs_are_skipped(self):
        records = [
            COGXEntity(external_system="z", external_id="n1", name="Alice"),
            COGXFact(
                external_system="z",
                external_id="f1",
                subject_ref="n1",
                predicate="mentions",
                object_ref="99999999-9999-9999-9999-999999999999",
            ),
        ]
        result = translate_records(records, "preserve")
        assert result.skipped_facts == 1
        batch = result.graph_batches[0]
        assert batch["edges"] == []
        # No Entity literally named by the dangling UUID.
        node_names = {getattr(node, "name", None) for node in batch["nodes"]}
        assert "99999999-9999-9999-9999-999999999999" not in node_names

    def test_graph_batches_are_bounded(self, monkeypatch):
        monkeypatch.setattr(loader, "BATCH_NODE_TARGET", 3)
        records = [
            COGXEntity(external_system="z", external_id=f"n{i}", name=f"Entity {i}")
            for i in range(8)
        ] + [
            COGXFact(
                external_system="z",
                external_id="f-cross",
                subject_ref="n0",
                predicate="knows",
                object_ref="n7",
            )
        ]
        result = translate_records(records, "preserve")
        assert len(result.graph_batches) > 1
        assert all(len(batch["nodes"]) <= 4 for batch in result.graph_batches)
        # The cross-batch fact's batch contains both endpoints.
        edge_batch = next(batch for batch in result.graph_batches if batch["edges"])
        source_id, target_id, _, _ = edge_batch["edges"][0]
        node_ids = {node.id for node in edge_batch["nodes"]}
        assert source_id in node_ids and target_id in node_ids


class TestFormatEmitters:
    NODES = [
        ("id-1", {"type": "Entity", "name": "Alice", "description": "A person"}),
        ("id-2", {"type": "Entity", "name": "Berlin", "tags": ["city", "capital"]}),
    ]
    EDGES = [("id-1", "id-2", "lives_in", {"edge_text": "Alice lives in Berlin"})]

    def test_json(self, tmp_path):
        destination = tmp_path / "graph.json"
        write_json(self.NODES, self.EDGES, destination)
        payload = json.loads(destination.read_text())
        assert len(payload["nodes"]) == 2
        assert payload["edges"][0]["relationship_name"] == "lives_in"
        assert payload["edges"][0]["edge_text"] == "Alice lives in Berlin"

    def test_graphml_is_valid_xml(self, tmp_path):
        destination = tmp_path / "graph.graphml"
        write_graphml(self.NODES, self.EDGES, destination)
        root = ET.parse(destination).getroot()
        namespace = "{http://graphml.graphdrawing.org/xmlns}"
        graph = root.find(f"{namespace}graph")
        assert len(graph.findall(f"{namespace}node")) == 2
        assert len(graph.findall(f"{namespace}edge")) == 1

    def test_cypher(self, tmp_path):
        destination = tmp_path / "graph.cypher"
        write_cypher(self.NODES, self.EDGES, destination)
        script = destination.read_text()
        # Indexed shared label keeps edge MATCH clauses off AllNodesScan plans.
        assert "CREATE INDEX IF NOT EXISTS FOR (n:CogneeNode) ON (n.id);" in script
        assert 'MERGE (n:CogneeNode {id: "id-1"}) SET n:`Entity`' in script
        assert "MERGE (a)-[r:`lives_in`]->(b)" in script
        # Non-scalar property serialized as JSON string, not raw repr.
        assert '"[\\"city\\", \\"capital\\"]"' in script


class TestImportModeValidation:
    def test_unknown_mode_rejected(self):
        try:
            Mem0Source([], mode="yolo")
            raise AssertionError("expected ValueError")
        except ValueError as error:
            assert "yolo" in str(error)
