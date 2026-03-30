from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PipelineContext:
    """Typed runtime context for pipeline tasks.

    Tasks that need runtime values accept this as an explicit parameter::

        async def add_data_points(data_points, ctx: PipelineContext = None):
            if ctx:
                user = ctx.user
                dataset = ctx.dataset

    The pipeline machinery passes ``ctx`` to any task whose signature
    includes a parameter named ``ctx`` with type ``PipelineContext``.
    """

    user: Any = None
    data_item: Any = None
    dataset: Any = None
    pipeline_name: Optional[str] = None
