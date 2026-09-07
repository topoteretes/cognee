"""Per-adapter capability probe for the improve loop (plan Part 5.6).

Feedback weights exist on the Ladybug and Neo4j adapters, truth state on
Ladybug alone. Everywhere else ``GraphDBInterface`` holds stubs that raise
``NotImplementedError``. Rather than letting a stage discover that by failing
into a swallowed warning, the orchestrator probes the adapter once per run and
hands the answer to every stage on ``ImproveRunInputs.capabilities``.

An adapter supports a method when its class overrides the interface's stub
(compared by function identity), or when it declares an explicit boolean class
attribute of the same ``supports_*`` name. ``supports_incremental_chunk_updates``
(PR #4874) is read the same way, so this is the adapters' only capability
surface.
"""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from cognee.shared.logging_utils import get_logger

logger = get_logger("improve.capabilities")

_FEEDBACK_WEIGHT_METHODS = ("set_node_feedback_weights", "set_edge_feedback_weights")
_TRUTH_STATE_METHODS = ("get_node_truth_state", "set_node_truth_state")


class GraphCapabilities(BaseModel):
    """What the run's graph adapter can do. Resolved once per run."""

    model_config = ConfigDict(frozen=True)

    supports_feedback_weights: bool
    supports_truth_state: bool
    supports_incremental_chunk_updates: bool = False
    adapter: Optional[str] = None

    @classmethod
    def assume_supported(cls, adapter: Optional[str] = None) -> "GraphCapabilities":
        """Fail-open answer used when the adapter could not be probed.

        A stage that runs against an unsupported backend reports ``errored`` —
        the behaviour before the probe existed — instead of being skipped on
        a guess.
        """
        return cls(
            supports_feedback_weights=True,
            supports_truth_state=True,
            supports_incremental_chunk_updates=False,
            adapter=adapter,
        )


def _overrides_interface(engine: Any, method_name: str) -> bool:
    """True when ``engine``'s class implements ``method_name`` beyond the interface stub."""
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    interface_method = getattr(GraphDBInterface, method_name, None)
    if isinstance(engine, GraphDBInterface):
        adapter_method = getattr(type(engine), method_name, None)
        if adapter_method is None:
            return False
        return adapter_method is not interface_method
    # Duck-typed engines (community adapters, test doubles): any callable counts.
    return callable(getattr(engine, method_name, None))


def _explicit_flag(engine: Any, flag_name: str) -> Optional[bool]:
    value = getattr(engine, flag_name, None)
    return value if isinstance(value, bool) else None


def _supports(engine: Any, flag_name: str, method_names: tuple) -> bool:
    explicit = _explicit_flag(engine, flag_name)
    if explicit is not None:
        return explicit
    return all(_overrides_interface(engine, name) for name in method_names)


def probe_graph_capabilities(engine: Any) -> GraphCapabilities:
    """Answer the improve loop's capability questions for one graph engine."""
    return GraphCapabilities(
        supports_feedback_weights=_supports(
            engine, "supports_feedback_weights", _FEEDBACK_WEIGHT_METHODS
        ),
        supports_truth_state=_supports(engine, "supports_truth_state", _TRUTH_STATE_METHODS),
        supports_incremental_chunk_updates=bool(
            getattr(engine, "supports_incremental_chunk_updates", False)
        ),
        adapter=type(engine).__name__,
    )


async def resolve_graph_capabilities(
    dataset_id: UUID, owner_id: Optional[UUID]
) -> GraphCapabilities:
    """Probe the graph adapter that serves ``dataset_id``.

    Enters the dataset's database scope only for the probe and leaves it
    before any stage runs (holding the scope's queue slot while a pipeline
    waits on the dataset lock can deadlock — SDK-483). Fails open: if the
    engine cannot be created, the run assumes support and lets the stages
    report their own errors.
    """
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine

    try:
        async with set_database_global_context_variables(dataset_id, owner_id):
            engine = await get_graph_engine()
            return probe_graph_capabilities(engine)
    except Exception as error:
        logger.warning("improve: graph capability probe failed, assuming support: %s", error)
        return GraphCapabilities.assume_supported()
