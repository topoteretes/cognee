"""Unit tests for live-API memory sources (mocked clients, no network)."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cognee.modules.migration.sources.live.graphiti import (
    GraphitiLiveSource,
    fetch_graphiti_snapshot,
)
from cognee.modules.migration.sources.live.letta import LettaLiveSource, fetch_letta_snapshot
from cognee.modules.migration.sources.live.mem0 import Mem0LiveSource
from cognee.modules.migration.sources.live.zep import ZepLiveSource, fetch_zep_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "live"


def collect(source):
    async def _collect():
        return [record async for record in source.records()]

    return asyncio.run(_collect())


def _load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestMem0LiveSource:
    def test_paginates_and_maps_memories(self):
        fixture = _load_fixture("mem0_responses.json")
        calls = []

        def get_all(filters=None, page=1, page_size=100):
            calls.append(page)
            return fixture["page1"] if page == 1 else fixture["page2"]

        client = SimpleNamespace(get_all=get_all)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "mem0", MagicMock())
            records = collect(Mem0LiveSource(client, filters={"user_id": "u1"}, page_size=1))

        assert calls == [1, 2]
        assert len(records) == 2
        assert {record.kind for record in records} == {"memory"}
        assert records[0].external_id == "mem-1"
        assert records[1].content == "Alice likes tea"

    def test_skips_items_without_content(self):
        client = SimpleNamespace(
            get_all=lambda **kwargs: {
                "results": [{"id": "1"}, {"id": "2", "memory": "kept"}],
                "next": None,
            }
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "mem0", MagicMock())
            records = collect(Mem0LiveSource(client, filters={"user_id": "u1"}))
        assert len(records) == 1
        assert records[0].content == "kept"

    def test_requires_entity_filters(self):
        client = SimpleNamespace(get_all=lambda **kwargs: {"results": []})
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "mem0", MagicMock())
            source = Mem0LiveSource(client)
            with pytest.raises(ValueError, match="entity-scoped filters"):
                collect(source)

    def test_replayable_snapshot(self):
        calls = []

        def get_all(**kwargs):
            calls.append(1)
            return {"results": [{"id": "1", "memory": "once"}], "next": None}

        client = SimpleNamespace(get_all=get_all)
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "mem0", MagicMock())
            source = Mem0LiveSource(client, filters={"user_id": "u1"})
            first = collect(source)
            second = collect(source)

        assert len(calls) == 1
        assert len(first) == len(second) == 1


class TestGraphitiLiveSource:
    def test_fetch_and_map_graph_snapshot(self, monkeypatch):
        fixture = _load_fixture("graphiti_responses.json")

        class FakeEpisodicNode:
            @staticmethod
            async def get_by_group_ids(driver, group_ids, limit, uuid_cursor=None):
                return [
                    SimpleNamespace(**fixture["episodes"][0], source=SimpleNamespace(value="text"))
                ]

        class FakeEntityNode:
            @staticmethod
            async def get_by_group_ids(driver, group_ids, limit, uuid_cursor=None):
                return [SimpleNamespace(**node) for node in fixture["nodes"]]

        class FakeEntityEdge:
            @staticmethod
            async def get_by_group_ids(driver, group_ids, limit, uuid_cursor=None):
                return [SimpleNamespace(**fixture["edges"][0])]

        fake_graphiti_core = SimpleNamespace(
            nodes=SimpleNamespace(EpisodicNode=FakeEpisodicNode, EntityNode=FakeEntityNode),
            edges=SimpleNamespace(EntityEdge=FakeEntityEdge),
        )
        monkeypatch.setitem(sys.modules, "graphiti_core", fake_graphiti_core)

        graphiti = SimpleNamespace(driver=object())
        records = collect(GraphitiLiveSource(graphiti, group_ids=["g1"]))

        kinds = [record.kind for record in records]
        assert kinds == ["episode", "entity", "entity", "fact"]
        assert records[-1].fact_text == "Alice lives in Berlin"

    def test_fetch_graphiti_snapshot_helper(self, monkeypatch):
        fixture = _load_fixture("graphiti_responses.json")

        class FakeNode:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def model_dump(self, mode="json"):
                return dict(self.__dict__)

        class FakeEpisodicNode:
            @staticmethod
            async def get_by_group_ids(driver, group_ids, limit, uuid_cursor=None):
                return [FakeNode(**fixture["episodes"][0], source="text")]

        class FakeEntityNode:
            @staticmethod
            async def get_by_group_ids(driver, group_ids, limit, uuid_cursor=None):
                return []

        class GroupsEdgesNotFoundError(Exception):
            pass

        class FakeEntityEdge:
            @staticmethod
            async def get_by_group_ids(driver, group_ids, limit, uuid_cursor=None):
                raise GroupsEdgesNotFoundError()

        fake_graphiti_core = SimpleNamespace(
            nodes=SimpleNamespace(EpisodicNode=FakeEpisodicNode, EntityNode=FakeEntityNode),
            edges=SimpleNamespace(EntityEdge=FakeEntityEdge),
        )
        monkeypatch.setitem(sys.modules, "graphiti_core", fake_graphiti_core)

        snapshot = asyncio.run(
            fetch_graphiti_snapshot(SimpleNamespace(driver=object()), group_ids=[""])
        )
        assert len(snapshot["episodes"]) == 1
        assert snapshot["edges"] == []


class TestZepLiveSource:
    def test_cursor_pagination_and_uuid_normalization(self, monkeypatch):
        node_batches = [
            [SimpleNamespace(uuid_="node-1", name="Berlin", labels=["Entity"], summary="A city")],
            [],
        ]
        edge_batches = [
            [
                SimpleNamespace(
                    uuid_="edge-1",
                    source_node_uuid="node-1",
                    target_node_uuid="node-2",
                    name="LOCATED_IN",
                    fact="Berlin is in Germany",
                    valid_at="2024-03-01T00:00:00Z",
                    episodes=["ep-zep-1", "ep-missing"],
                )
            ],
            [],
        ]
        node_calls = []
        edge_calls = []

        def get_nodes(user_id, limit, uuid_cursor=None):
            node_calls.append(uuid_cursor)
            return node_batches.pop(0)

        def get_edges(user_id, limit, uuid_cursor=None):
            edge_calls.append(uuid_cursor)
            return edge_batches.pop(0)

        missing_episode = SimpleNamespace(
            uuid_="ep-missing",
            name="extra",
            content="Referenced episode",
            created_at="2024-03-02T00:00:00Z",
        )

        graph = SimpleNamespace(
            episode=SimpleNamespace(
                get_by_user_id=lambda user_id, lastn: SimpleNamespace(
                    episodes=[
                        SimpleNamespace(
                            uuid_="ep-zep-1",
                            name="session",
                            content="User asked about Berlin",
                            created_at="2024-03-01T00:00:00Z",
                            group_id="sess-1",
                        )
                    ]
                ),
                get=lambda uuid_: missing_episode,
            ),
            node=SimpleNamespace(get_by_user_id=get_nodes),
            edge=SimpleNamespace(get_by_user_id=get_edges),
        )
        client = SimpleNamespace(graph=graph)

        monkeypatch.setitem(sys.modules, "zep_cloud", MagicMock())
        records = collect(ZepLiveSource(client, user_id="user-1", page_size=1))

        assert node_calls == [None, "node-1"]
        assert edge_calls == [None, "edge-1"]
        kinds = [record.kind for record in records]
        assert kinds.count("episode") == 2
        assert "entity" in kinds
        assert "fact" in kinds

    def test_fetch_zep_snapshot_requires_scope(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "zep_cloud", MagicMock())
        with pytest.raises(ValueError, match="user_id or graph_id"):
            asyncio.run(fetch_zep_snapshot(SimpleNamespace(graph=SimpleNamespace())))


class TestLettaLiveSource:
    def test_maps_agent_state(self, monkeypatch):
        agent = SimpleNamespace(id="agent-1", name="assistant")
        block = SimpleNamespace(id="blk1", label="persona", value="I am helpful", limit=2000)
        messages = [
            SimpleNamespace(
                message_type="user_message",
                content="hello",
                date="2024-01-01T00:00:00Z",
            ),
            SimpleNamespace(
                message_type="assistant_message",
                content=[{"type": "text", "text": "hi there"}],
                date=None,
            ),
            SimpleNamespace(message_type="system_message", content="ignored", date=None),
        ]
        passage = SimpleNamespace(id="p1", text="archived note", created_at="2024-01-02T00:00:00Z")

        client = SimpleNamespace(
            agents=SimpleNamespace(
                list=lambda: [agent],
                retrieve=lambda agent_id: agent,
                blocks=SimpleNamespace(list=lambda agent_id: [block]),
                messages=SimpleNamespace(list=lambda agent_id, order="asc": messages),
                passages=SimpleNamespace(list=lambda agent_id: [passage]),
            )
        )

        monkeypatch.setitem(sys.modules, "letta_client", MagicMock())
        records = collect(LettaLiveSource(client))

        kinds = [record.kind for record in records]
        assert kinds.count("memory_block") == 1
        assert kinds.count("episode") == 1
        assert kinds.count("document") == 1

        episode = next(record for record in records if record.kind == "episode")
        assert len(episode.turns) == 2

    def test_fetch_letta_snapshot_with_explicit_agent_ids(self, monkeypatch):
        client = SimpleNamespace(
            agents=SimpleNamespace(
                retrieve=lambda agent_id: SimpleNamespace(name="bot"),
                blocks=SimpleNamespace(list=lambda agent_id: []),
                messages=SimpleNamespace(list=lambda agent_id, order="asc": []),
                passages=SimpleNamespace(list=lambda agent_id: []),
            )
        )
        monkeypatch.setitem(sys.modules, "letta_client", MagicMock())
        snapshot = asyncio.run(fetch_letta_snapshot(client, agent_ids=["a1"]))
        assert snapshot["agents"][0]["name"] == "bot"


class TestReplayable:
    def test_second_records_call_does_not_refetch(self, monkeypatch):
        fetch_calls = []

        async def fake_fetch(
            client, user_id=None, graph_id=None, episode_lastn=10000, page_size=100
        ):
            fetch_calls.append(1)
            return _load_fixture("zep_responses.json")

        monkeypatch.setattr(
            "cognee.modules.migration.sources.live.zep.fetch_zep_snapshot",
            fake_fetch,
        )
        source = ZepLiveSource(SimpleNamespace(), user_id="u1")
        collect(source)
        collect(source)
        assert len(fetch_calls) == 1
