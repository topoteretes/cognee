from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


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

    Shared-mutable-state behavior (IMPORTANT)
    -----------------------------------------
    The worker pipeline derives a per-item context by calling
    ``dataclasses.replace(shared_ctx, data_item=origin)`` so each in-flight
    item gets its own ``ctx.data_item``. ``dataclasses.replace`` performs a
    **shallow copy**: every field that is not explicitly overridden is copied
    by reference, so all per-item contexts produced from the same
    ``shared_ctx`` share the *same* underlying mutable objects (``extras``,
    ``_provenance_visited``, etc.).

    ``_provenance_visited`` relies on this on purpose: a single shared set
    lets ``_stamp_provenance`` deduplicate visits across workers and across
    pipeline stages within one run.

    ``extras`` is also shared by reference across per-item contexts (same
    shallow-copy reason). It is meant for caller-supplied configuration read
    by tasks, not as a per-item scratchpad — treat it as read-mostly. If
    tasks need per-item state, store it in returned values or in a per-item
    field instead of mutating ``extras`` mid-pipeline.

    Warning for future maintainers: any new mutable field added to this
    class must either be

        (a) immutable (e.g. ``str``, ``int``, ``tuple``, ``frozenset``), or
        (b) explicitly cloned by the pipeline executor before per-item work
            (so each item gets its own copy), or
        (c) documented here as intentionally shared by reference across
            all in-flight items (like ``_provenance_visited``).

    Otherwise authors will reasonably assume per-item isolation and silently
    introduce cross-item state corruption.
    """

    user: Any = None
    data_item: Any = None
    dataset: Any = None
    pipeline_name: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # Internal: persisted across tasks so _stamp_provenance skips
    # DataPoints that were already walked in earlier pipeline stages.
    # NOTE: intentionally shared by reference across all per-item contexts
    # produced via ``dataclasses.replace`` within a single pipeline run --
    # this is what enables visit-deduplication across workers and stages.
    _provenance_visited: Set[int] = field(default_factory=set, repr=False)
