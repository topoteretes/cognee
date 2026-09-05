"""The frozen argument bundle handed to every stage (plan Part 5.3).

Assembled once by the orchestrator after the dataset is resolved and the
adapter probed. Nothing mutates it during a run. It is an argument bundle,
not a context: ``OperationContext`` (``record_operation``) stays the mutable
context for the ``pipeline_runs`` row. It has no ``run_in_background`` field
because the runner owns background mode and a stage never asks.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Tuple
from uuid import UUID

from .capabilities import GraphCapabilities
from .config import ImproveConfig

# memify() knobs that improve() forwards untouched to the enrichment stage.
MEMIFY_PASSTHROUGH_KEYS = (
    "extraction_tasks",
    "enrichment_tasks",
    "data",
    "node_type",
    "vector_db_config",
    "graph_db_config",
)


@dataclass(frozen=True)
class ImproveRunInputs:
    user: Any
    dataset_id: UUID
    dataset: Any  # the resolved Dataset row (id, name, owner_id)
    session_ids: Tuple[str, ...]
    config: ImproveConfig
    capabilities: GraphCapabilities
    node_name: Optional[List[str]] = None
    feedback_alpha: float = 0.1
    build_global_context_index: bool = False
    build_truth_subspace: bool = False
    # Caller-supplied memify overrides (extraction_tasks, enrichment_tasks,
    # data, node_type, vector_db_config, graph_db_config). Read-only.
    memify_kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.memify_kwargs, MappingProxyType):
            object.__setattr__(self, "memify_kwargs", MappingProxyType(dict(self.memify_kwargs)))
        if not isinstance(self.session_ids, tuple):
            object.__setattr__(self, "session_ids", tuple(self.session_ids or ()))

    @property
    def has_sessions(self) -> bool:
        return bool(self.session_ids)

    @property
    def session_id_list(self) -> List[str]:
        return list(self.session_ids)

    @property
    def has_custom_memify_tasks(self) -> bool:
        """True when the caller supplied its own memify tasks or data."""
        return any(
            self.memify_kwargs.get(key) for key in ("extraction_tasks", "enrichment_tasks", "data")
        )
