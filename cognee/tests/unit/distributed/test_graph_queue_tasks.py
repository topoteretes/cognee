import sys
from types import SimpleNamespace

import pytest

import distributed.tasks.queued_add_nodes as qan_module
import distributed.tasks.queued_add_edges as qae_module
from distributed.tasks.queued_add_edges import queued_add_edges
from distributed.tasks.queued_add_nodes import queued_add_nodes


class _FakePut:
    def __init__(self):
        self.items = []

    async def aio(self, item):
        self.items.append(item)


class _FakeGRPCError(Exception):
    """Stand-in for grpclib.GRPCError so the split path is testable without grpclib."""


class _RaisingPut:
    """Raises a (fake) grpc error on the first put only, records the rest.

    Lets us exercise the oversized-batch split without a real Modal/grpc backend.
    """

    def __init__(self):
        self.items = []
        self._raised = False

    async def aio(self, item):
        if not self._raised:
            self._raised = True
            raise _FakeGRPCError()
        self.items.append(item)


def _install_fake_graph_queue(monkeypatch, put=None):
    put = put or _FakePut()
    fake_queue = SimpleNamespace(put=put)
    monkeypatch.setitem(
        sys.modules,
        "distributed.queues",
        SimpleNamespace(add_nodes_and_edges_queue=fake_queue),
    )
    return put.items


@pytest.mark.asyncio
async def test_queued_add_nodes_defaults_none_provenance(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    await queued_add_nodes(["node-1"])

    assert queued_items == [(["node-1"], [], None, None)]


@pytest.mark.asyncio
async def test_queued_add_edges_defaults_none_provenance(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    await queued_add_edges([("source", "target", "REL", {})])

    assert queued_items == [([], [("source", "target", "REL", {})], None, None)]


@pytest.mark.asyncio
async def test_queued_add_nodes_carries_provenance_in_payload(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    await queued_add_nodes(["node-1"], source_ref_key="source-ref", pipeline_run_id="run-id")

    assert queued_items == [(["node-1"], [], "source-ref", "run-id")]


@pytest.mark.asyncio
async def test_queued_add_edges_carries_provenance_in_payload(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch)

    await queued_add_edges(
        [("source", "target", "REL", {})], source_ref_key="source-ref", pipeline_run_id="run-id"
    )

    assert queued_items == [([], [("source", "target", "REL", {})], "source-ref", "run-id")]


@pytest.mark.asyncio
async def test_queued_add_nodes_splits_and_preserves_provenance_on_grpc_error(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch, put=_RaisingPut())
    monkeypatch.setattr(
        qan_module, "_is_grpc_error", lambda error: isinstance(error, _FakeGRPCError)
    )

    await queued_add_nodes(["a", "b", "c", "d"], source_ref_key="sr", pipeline_run_id="run")

    # The full batch's put raised, so it splits into halves — each still tagged
    # with the same provenance.
    assert queued_items == [
        (["a", "b"], [], "sr", "run"),
        (["c", "d"], [], "sr", "run"),
    ]


@pytest.mark.asyncio
async def test_queued_add_edges_splits_and_preserves_provenance_on_grpc_error(monkeypatch):
    queued_items = _install_fake_graph_queue(monkeypatch, put=_RaisingPut())
    monkeypatch.setattr(
        qae_module, "_is_grpc_error", lambda error: isinstance(error, _FakeGRPCError)
    )

    edges = [("a", "b", "R", {}), ("c", "d", "R", {})]
    await queued_add_edges(edges, source_ref_key="sr", pipeline_run_id="run")

    assert queued_items == [
        ([], [edges[0]], "sr", "run"),
        ([], [edges[1]], "sr", "run"),
    ]
