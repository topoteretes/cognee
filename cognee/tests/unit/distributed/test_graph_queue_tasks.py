import sys
from types import SimpleNamespace

import pytest

from distributed.tasks.queued_add_edges import queued_add_edges
from distributed.tasks.queued_add_nodes import queued_add_nodes


class _FakePut:
    def __init__(self):
        self.items = []

    async def aio(self, item):
        self.items.append(item)


def _install_fake_graph_queue(monkeypatch):
    fake_put = _FakePut()
    fake_queue = SimpleNamespace(put=fake_put)
    monkeypatch.setitem(
        sys.modules,
        "distributed.queues",
        SimpleNamespace(add_nodes_and_edges_queue=fake_queue),
    )
    return fake_put.items


@pytest.mark.asyncio
async def test_queued_add_nodes_accepts_empty_provenance_kwargs(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    await queued_add_nodes(["node-1"], source_ref_key=None, pipeline_run_id=None)

    assert queued_items == [(["node-1"], [])]


@pytest.mark.asyncio
async def test_queued_add_edges_accepts_empty_provenance_kwargs(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    await queued_add_edges([("source", "target", "REL", {})], source_ref_key=None)

    assert queued_items == [([], [("source", "target", "REL", {})])]


@pytest.mark.asyncio
async def test_queued_add_nodes_rejects_real_provenance(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    with pytest.raises(NotImplementedError, match="graph provenance payloads"):
        await queued_add_nodes(["node-1"], source_ref_key="source-ref")

    assert queued_items == []


@pytest.mark.asyncio
async def test_queued_add_edges_rejects_real_provenance(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    with pytest.raises(NotImplementedError, match="graph provenance payloads"):
        await queued_add_edges([("source", "target", "REL", {})], pipeline_run_id="run-id")

    assert queued_items == []
