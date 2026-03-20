from uuid import UUID
from typing import Union, Optional, List, Type, Any

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from typing_extensions import TypedDict


class ImproveKwargs(TypedDict, total=False):
    """Power-user overrides for improve(). Most users never need these."""

    extraction_tasks: list
    enrichment_tasks: list
    data: Any
    node_type: Type
    user: object  # User context (resolved internally when None)
    vector_db_config: dict
    graph_db_config: dict


async def improve(
    dataset: Union[str, UUID] = "main_dataset",
    *,
    run_in_background: bool = False,
    node_name: Optional[List[str]] = None,
    **kwargs: Unpack[ImproveKwargs],
):
    """Enrich an existing knowledge graph with additional context and rules.

    This is a memory-oriented alias for ``cognee.memify()``.  The most common
    parameters are explicit keyword arguments; power-user options can be passed
    via ``ImproveKwargs`` (see class definition for available keys).

    Args:
        dataset: Dataset name or UUID to process.
        run_in_background: Run processing asynchronously.
        node_name: Filter graph to specific named entities.
        **kwargs: Additional options — see ``ImproveKwargs``.

    Returns:
        Pipeline run info (same as ``cognee.memify()``).
    """
    from cognee.modules.memify import memify

    # Resolve default node_type here to avoid import at module level
    if "node_type" not in kwargs or kwargs.get("node_type") is None:
        from cognee.modules.engine.models.node_set import NodeSet

        kwargs["node_type"] = NodeSet

    return await memify(
        dataset=dataset,
        node_name=node_name,
        run_in_background=run_in_background,
        **kwargs,
    )
