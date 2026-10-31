"""The frozen argument bundle handed to every stage (plan Part 5.3).

Assembled once by the orchestrator after the dataset is resolved and the
adapter probed. Nothing mutates it during a run. It is an argument bundle,
not a context: ``OperationContext`` (``record_operation``) stays the mutable
context for the ``pipeline_runs`` row. It has no ``run_in_background`` field
because ``improve()`` owns background mode and a stage never asks.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any
from uuid import UUID

from .capabilities import GraphCapabilities
from .config import ImproveConfig
from .constants import DEFAULT_FEEDBACK_ALPHA


@dataclass(frozen=True)
class ImproveRunInputs:
    user: Any
    dataset_id: UUID
    dataset: Any  # the resolved Dataset row (id, name, owner_id)
    session_ids: tuple[str, ...]
    config: ImproveConfig
    capabilities: GraphCapabilities
    # The run's own operation-record id (``pipeline_runs.pipeline_run_id``).
    improve_operation_id: UUID | None = None
    feedback_alpha: float = DEFAULT_FEEDBACK_ALPHA
    build_global_context_index: bool = False
    build_truth_subspace: bool = False
    review_conflicts: bool = True
    # Deprecated (SDK-1115): the caller's improve(extraction_tasks=...,
    # enrichment_tasks=...) and the options that scope them (data, node_type,
    # node_name, vector_db_config, graph_db_config). None unless custom tasks
    # were given; then the custom_tasks stage runs them. Read-only.
    legacy_memify: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.legacy_memify is not None and not isinstance(self.legacy_memify, MappingProxyType):
            object.__setattr__(self, "legacy_memify", MappingProxyType(dict(self.legacy_memify)))
        if not isinstance(self.session_ids, tuple):
            object.__setattr__(self, "session_ids", tuple(self.session_ids or ()))

    def with_capabilities(self, capabilities: GraphCapabilities) -> "ImproveRunInputs":
        """A copy carrying the probed adapter capabilities.

        The probe leases the graph engine (and, with backend access control
        on, a dataset-queue slot), so the orchestrator resolves it only AFTER
        winning the lock claim — a lock loser must not pay for a probe that
        only stages 1 and 7 read.
        """
        return replace(self, capabilities=capabilities)

    @property
    def dataset_name(self) -> str | None:
        """The resolved dataset's name — for reporting only, never for resolution."""
        return getattr(self.dataset, "name", None)

    @property
    def has_sessions(self) -> bool:
        return bool(self.session_ids)

    @property
    def session_id_list(self) -> list[str]:
        return list(self.session_ids)
