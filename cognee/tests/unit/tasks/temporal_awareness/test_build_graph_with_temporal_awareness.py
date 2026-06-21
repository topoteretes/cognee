import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class FakeEpisodeType:
    text = "text"


class FakeGraphiti:
    def __init__(self, url, user, password):
        self.url = url
        self.user = user
        self.password = password
        self.episodes = []

    async def build_indices_and_constraints(self):
        self.indices_built = True

    async def add_episode(self, **kwargs):
        self.episodes.append(kwargs)


def load_temporal_module(monkeypatch):
    graphiti_module = types.ModuleType("graphiti_core")
    graphiti_module.Graphiti = FakeGraphiti
    nodes_module = types.ModuleType("graphiti_core.nodes")
    nodes_module.EpisodeType = FakeEpisodeType
    monkeypatch.setitem(sys.modules, "graphiti_core", graphiti_module)
    monkeypatch.setitem(sys.modules, "graphiti_core.nodes", nodes_module)
    sys.modules.pop("cognee.tasks.temporal_awareness", None)
    sys.modules.pop("cognee.tasks.temporal_awareness.build_graph_with_temporal_awareness", None)
    return importlib.import_module(
        "cognee.tasks.temporal_awareness.build_graph_with_temporal_awareness"
    )


@pytest.mark.asyncio
async def test_build_graph_with_temporal_awareness_reads_file_uris(monkeypatch, tmp_path):
    module = load_temporal_module(monkeypatch)
    file_path = tmp_path / "temporal notes.txt"
    file_path.write_text("remember this", encoding="utf-8")
    data = [SimpleNamespace(raw_data_location=file_path.as_uri())]

    graphiti = await module.build_graph_with_temporal_awareness(data)

    assert graphiti.indices_built is True
    assert graphiti.episodes[0]["episode_body"] == "remember this"
    assert graphiti.episodes[0]["source"] == "text"
