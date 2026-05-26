from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class PipelineContext:
    """Typed runtime context for pipeline tasks.

    Tasks that need runtime values accept this as an explicit parameter::

        async def add_data_points(data_points, ctx: PipelineContext = None):
            if ctx:
                user = ctx.user
                dataset = ctx.dataset
                custom_val = ctx.extras.get("my_key")

    The pipeline machinery passes ``ctx`` to any task whose signature
    includes a parameter named ``ctx`` (matched by name, not by type annotation).

    Custom pipelines can store additional state in ``extras``.
    """

    user: Any = None
    data_item: Any = None
    dataset: Any = None
    pipeline_name: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # Internal: persisted across tasks so _stamp_provenance skips
    # DataPoints that were already walked in earlier pipeline stages.
    _provenance_visited: Set[int] = field(default_factory=set, repr=False)

    # Ordered list of unique task names in the order they were first seen.
    # Used to derive DataPoint.topological_rank (1-based index into this list)
    # so the visualization can lay nodes out in pipeline order. A task that
    # runs N times because an upstream task streams N results still gets the
    # same rank — rank is "position in pipeline", not "invocation count".
    task_sequence: List[str] = field(default_factory=list)
