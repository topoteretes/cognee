"""Unit tests for the memory-migration module (CMIF, sources, loader, formats).

All tests are pure: no databases, no LLM calls.
"""

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from cognee.modules.migration.cmif import (
    CMIFArchiveWriter,
    CMIFDocument,
    CMIFEntity,
    CMIFEpisode,
    CMIFFact,
    CMIFMemory,
    CMIFMemoryBlock,
    CMIFTurn,
    parse_timestamp,
    read_archive,
    read_manifest,
)
from cognee.modules.migration.formats import write_cypher, write_graphml, write_json
from cognee.modules.migration.loader import record_data_id, translate_records
from cognee.modules.migration.sources import (
    CMIFArchiveSource,
    GraphitiSource,
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

    def test_invalid_values(self):
        assert parse_timestamp(None) is None
        assert parse_timestamp("not-a-date") is None


class TestCMIFArchive:
    def _sample_records(self):
        return [
            CMIFDocument(external_system="test", external_id="d1", content="hello"),
            CMIFEpisode(
                external_system="test",
                external_id="e1",
                turns=[CMIFTurn(role="user", content="hi")],
            ),
            CMIFEntity(external_system="test", external_id="n1", name="Alice"),
            CMIFFact(
                external_system="test",
                external_id="f1",
                subject_ref="n1",
                predicate="works_at",
                object_ref="Acme",
                valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            CMIFMemory(external_system="test", external_id="m1", content="likes tea"),
            CMIFMemoryBlock(external_system="test", external_id="b1", label="persona", value="x"),
        ]

    def test_roundtrip(self, tmp_path):
        archive_dir = tmp_path / "archive"
        records = self._sample_records()
        with CMIFArchiveWriter(archive_dir, source_system="test") as writer:
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
        with CMIFArchiveWriter(archive_dir, source_system="test") as writer:
            for record in self._sample_records():
                writer.write(record)

        source = CMIFArchiveSource(archive_dir)
        assert source.source_system == "test"
        assert len(collect(source)) == 6


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
            CMIFMemory(external_system="mem0", external_id="m1", content="likes tea"),
            CMIFEpisode(
                external_system="zep",
                external_id="e1",
                turns=[
                    CMIFTurn(
                        role="user",
                        content="hello",
                        occurred_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    )
                ],
            ),
            CMIFEntity(
                external_system="zep",
                external_id="n1",
                name="Alice",
                entity_type="Person",
                description="A person",
                aliases=["Alice Smith"],
            ),
            CMIFFact(
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
        record = CMIFMemory(external_system="mem0", external_id="m1", content="x")
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
            CMIFEntity(external_system="a", external_id="x1", name="Alice"),
            CMIFEntity(external_system="b", external_id="x2", name="Alice"),
        ]
        batch = translate_records(records, "preserve").graph_batches[0]
        entities = [node for node in batch["nodes"] if type(node).__name__ == "Entity"]
        assert len(entities) == 1


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
        assert 'MERGE (n:`Entity` {id: "id-1"})' in script
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
