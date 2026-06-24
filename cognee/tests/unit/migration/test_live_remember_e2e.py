"""E2E tests: remember() dispatch with live MemorySource importers (mocked fetch)."""

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cognee.api.v1.remember.remember import remember
from cognee.modules.migration.sources.live.graphiti import GraphitiLiveSource
from cognee.modules.migration.sources.live.letta import LettaLiveSource
from cognee.modules.migration.sources.live.mem0 import Mem0LiveSource
from cognee.modules.migration.sources.live.zep import ZepLiveSource
from cognee.tests.unit.migration.test_import_source import _summary_items, install_sinks

serve_state = importlib.import_module("cognee.api.v1.serve.state")
shared_utils = importlib.import_module("cognee.shared.utils")

FIXTURES = Path(__file__).parent / "fixtures" / "live"


def _load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(shared_utils, "send_telemetry", lambda *args, **kwargs: None)


class TestRememberLiveSources:
    def test_remember_mem0_live_re_derive(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        client = SimpleNamespace(
            get_all=lambda **kwargs: {
                "results": [{"id": "m1", "memory": "likes tea", "user_id": "u1"}],
                "next": None,
            }
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "mem0", MagicMock())
            result = asyncio.run(remember(Mem0LiveSource(client, filters={"user_id": "u1"}), "ds"))

        assert len(sinks.remember_calls) == 1
        assert sinks.pipeline_calls == []
        (summary,) = _summary_items(result)
        assert summary["source_system"] == "mem0"
        assert summary["mode"] == "re-derive"
        assert summary["record_counts"]["memory"] == 1

    def test_remember_graphiti_live_hybrid(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        fixture = _load_fixture("graphiti_responses.json")

        async def fake_fetch(graphiti, group_ids=None, page_size=500):
            return fixture

        monkeypatch.setattr(
            "cognee.modules.migration.sources.live.graphiti.fetch_graphiti_snapshot",
            fake_fetch,
        )
        result = asyncio.run(remember(GraphitiLiveSource(SimpleNamespace(driver=object())), "ds"))

        assert len(sinks.pipeline_calls) == 1
        assert len(sinks.remember_calls) == 1
        (summary,) = _summary_items(result)
        assert summary["source_system"] == "graphiti"
        assert summary["mode"] == "hybrid"
        assert summary["graph_nodes"] == 3  # Alice, Berlin, and Person entity type
        assert summary["graph_edges"] == 1

    def test_remember_zep_live_preserve(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        fixture = _load_fixture("zep_responses.json")

        async def fake_fetch(client, **kwargs):
            return fixture

        monkeypatch.setattr(
            "cognee.modules.migration.sources.live.zep.fetch_zep_snapshot",
            fake_fetch,
        )
        source = ZepLiveSource(SimpleNamespace(), user_id="u1", mode="preserve")
        result = asyncio.run(remember(source, "ds"))

        assert len(sinks.pipeline_calls) == 1
        assert len(sinks.add_calls) == 1
        assert sinks.remember_calls == []
        (summary,) = _summary_items(result)
        assert summary["source_system"] == "zep"
        assert summary["mode"] == "preserve"

    def test_remember_letta_live_re_derive(self, monkeypatch):
        sinks = install_sinks(monkeypatch)
        fixture = _load_fixture("letta_responses.json")

        async def fake_fetch(client, agent_ids=None):
            return fixture

        monkeypatch.setattr(
            "cognee.modules.migration.sources.live.letta.fetch_letta_snapshot",
            fake_fetch,
        )
        result = asyncio.run(remember(LettaLiveSource(SimpleNamespace()), "ds"))

        assert len(sinks.remember_calls) == 1
        (summary,) = _summary_items(result)
        assert summary["source_system"] == "letta"
        assert summary["mode"] == "re-derive"
        assert summary["record_counts"]["memory_block"] == 1
        assert summary["record_counts"]["episode"] == 1
        assert summary["record_counts"]["document"] == 1
