"""Optional storage writers supplied by plugins for a concrete adapter pair."""

from collections.abc import Callable
from typing import Protocol


class BatchWriter(Protocol):
    vector: object

    async def write_graph(
        self, nodes, edges, *, custom_edges, node_refs, edge_refs, default_ref, pipeline_run_id
    ) -> None: ...


_factories: dict[str, Callable[[object, object], BatchWriter | None]] = {}


def register_batch_writer_factory(
    name: str, factory: Callable[[object, object], BatchWriter | None]
) -> None:
    """Register once at plugin startup; repeating the same registration is harmless.

    A factory returns None for an unsupported pair or when its opt-in is disabled.
    Writers retain graph-before-vector ordering and the adapters' rollback contract.
    """
    _factories[name] = factory


def get_batch_writer(graph, vector) -> BatchWriter | None:
    writers = [writer for factory in _factories.values() if (writer := factory(graph, vector))]
    if len(writers) > 1:
        raise RuntimeError("Multiple batch writers registered for the same adapter pair")
    return writers[0] if writers else None
