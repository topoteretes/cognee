"""What a run and the workers that execute some of its data items tell each other.

A :class:`Shard` is what a worker is asked to do and a :class:`ShardResult` what
it answers. Neither is sent anywhere: both are read from and written to the
shard's rows (``shard_store``). Of a shard only the run's description travels
as JSON (:class:`PipelineRunSpec`), in a form derived from its field
declarations, so a field is declared once.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal
from uuid import UUID

from pydantic import TypeAdapter
from typing_extensions import Self

from cognee.modules.operations.usage_accumulator import get_active_operation_usage
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
)

# How a data item ended: the ``status`` of the run info its result entry carries.
ItemStatus = Literal["PipelineRunCompleted", "PipelineRunAlreadyCompleted", "PipelineRunErrored"]
COMPLETED: ItemStatus = "PipelineRunCompleted"
ALREADY_COMPLETED: ItemStatus = "PipelineRunAlreadyCompleted"
ERRORED: ItemStatus = "PipelineRunErrored"

_RUN_INFO = {
    COMPLETED: PipelineRunCompleted,
    ALREADY_COMPLETED: PipelineRunAlreadyCompleted,
    ERRORED: PipelineRunErrored,
}


@cache
def _adapter(message_type: type) -> TypeAdapter:
    return TypeAdapter(message_type)


@dataclass(frozen=True, kw_only=True)
class RunSettings:
    """What every data item of one pipeline run is run with, in any process."""

    pipeline_name: str
    pipeline_id: UUID
    pipeline_run_id: UUID
    incremental_loading: bool
    data_cache: bool
    # All of a run's items are scheduled at once; one process runs at most this
    # many of them at a time.
    data_per_batch: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class PipelineRunSpec(RunSettings):
    """What another process needs to run data items of a run it does not own:
    the run's settings, and how to find its dataset, its user and its tasks.

    Stored as JSON on each of the run's shards for the shard's worker to read.
    """

    dataset_id: UUID
    user_id: UUID
    # The name the run's task lists are registered under, and the arguments they
    # were built with (see ``task_resolvers``).
    task_resolver: str
    task_resolver_args: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return _adapter(type(self)).dump_python(self, mode="json")

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Self:
        return _adapter(cls).validate_python(raw)

    def survives_json(self) -> bool:
        """Whether this spec reads back equal after a trip through JSON text.

        ``extras`` and the resolver's arguments are the caller's: a value JSON
        cannot carry (an object, NaN) or carries differently (a set, a tuple, a
        non-string key) must keep the run in its own process.
        """
        adapter = _adapter(type(self))
        try:
            return adapter.validate_json(adapter.dump_json(self, warnings=False)) == self
        except (ValueError, RecursionError):
            return False


def survives_json(value: Any) -> bool:
    """Whether ``value`` reads back equal after a trip through JSON text.

    What JSON cannot carry (an object, NaN) or carries differently (a set, a
    tuple, a non-string key) does not.
    """
    try:
        return json.loads(json.dumps(value, allow_nan=False)) == value
    except (TypeError, ValueError, RecursionError):
        return False


@dataclass(frozen=True, kw_only=True)
class ShardItem:
    """One item of a shard: a ``Data`` row of the run's dataset, named by its id
    and read by the worker, or a value that is no row, carried as it is (it
    survives JSON)."""

    data_id: UUID | None = None
    value: Any = None


@dataclass(frozen=True, kw_only=True)
class Shard:
    """What one worker is asked to do: which run, and which of its items.

    Only this package reads what is in a shard. Whatever starts a worker (a
    ``WorkerBackend``) is handed the shard's id and nothing else, and the worker
    reads the shard from its rows. So what a worker is asked to do can change
    without any backend changing.
    """

    spec: PipelineRunSpec
    # By their position in the shard, which is what an item's outcome names.
    items: dict[int, ShardItem]

    @property
    def pipeline_run_id(self) -> UUID:
        return self.spec.pipeline_run_id

    @property
    def dataset_id(self) -> UUID:
        return self.spec.dataset_id


@dataclass
class ItemOutcome:
    """How one item of a shard ended: what its item row holds of a ``run_tasks``
    result entry. Its status, and for a failed item the class name and the
    scrubbed message of the failure. Nothing the item produced.

    ``index`` is the item's position in its shard. ``data_id`` is the ``Data``
    row the entry names: the item itself when it is one, else the row its tasks
    stored it as, if any."""

    status: ItemStatus
    index: int
    data_id: UUID | None = None
    error_class: str | None = None
    error_message: str | None = None

    @classmethod
    def from_entry(cls, entry: dict[str, Any], index: int) -> Self:
        run_info = entry["run_info"]
        return cls(
            status=run_info.status,
            index=index,
            data_id=entry["data_id"],
            error_class=getattr(run_info, "error_class", None),
            error_message=getattr(run_info, "error_message", None),
        )

    def to_entry(self, run: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
        """The result entry of this outcome. ``run`` is what every run info names
        (``pipeline_run_id``, ``dataset_id``, ``dataset_name``); ``source`` is
        what an entry says about where its item came from (its name, location
        and label), which the run's owner reads from the data item itself."""
        if self.status not in _RUN_INFO:
            raise ValueError(f"Unknown data item status {self.status!r}")
        entry: dict[str, Any] = {"data_id": self.data_id, **source}
        if self.status != ERRORED:
            entry["run_info"] = _RUN_INFO[self.status](**run)
            return entry
        # run_tasks names a run's error after the class of its first failed
        # item. The original exception stayed in the other process, so this one
        # carries its class name and its already scrubbed message.
        error = type(self.error_class or "DataItemError", (Exception,), {})(
            self.error_message or ""
        )
        entry["run_info"] = PipelineRunErrored(
            **run,
            payload=repr(error),
            error_class=self.error_class,
            error_message=self.error_message,
        )
        entry["error"] = error
        return entry


@dataclass
class ShardResult:
    """What one worker reports for the data items it ran."""

    outcomes: list[ItemOutcome]
    tokens_in: int = 0
    tokens_out: int = 0
    embedding_tokens: int = 0


def count_usage(shard_result: ShardResult) -> None:
    """Add what a shard's worker used to the active usage scope, which is the
    run's own inside ``run_tasks``. Once per result: it adds each time."""
    usage = get_active_operation_usage()
    if usage is not None:
        usage.add(shard_result.tokens_in, shard_result.tokens_out)
        usage.add_embedding(shard_result.embedding_tokens)
