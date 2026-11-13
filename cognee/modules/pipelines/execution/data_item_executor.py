"""Where the data items of one pipeline run execute.

``run_tasks`` owns a run: its record, its dataset context, its rollback and its
terminal status. By default it also runs every data item itself. A
:class:`DataItemExecutor`, registered for the process or handed to a call, may
run them in other processes instead, while the run stays one logical run with
one owner.

Only a run whose task resolver is *portable* can leave the process: a resolver
that names a registered factory and the JSON arguments another process rebuilds
it from (see ``task_resolvers``). Everything else runs in process as before.

Wherever an item runs, ``DataItemRunner`` (``run_tasks``) is what runs it. What
crosses the process boundary is in ``messages``.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from cognee.context_global_variables import backend_access_control_enabled
from cognee.context_global_variables import embedding_config as context_embedding_config
from cognee.context_global_variables import graph_db_config as context_graph_db_config
from cognee.context_global_variables import llm_config as context_llm_config
from cognee.context_global_variables import vector_db_config as context_vector_db_config
from cognee.infrastructure.llm.exceptions import raise_if_budget_exhausted_record
from cognee.modules.data.models import Data
from cognee.modules.pipelines.models.DataItemStatus import is_item_stamped
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored

from .messages import PipelineRunSpec, RunSettings


@dataclass
class ItemRun:
    """What ``run_tasks`` hands an executor for one run.

    ``run_in_process`` runs items the ordinary way and returns what
    ``run_tasks`` gathers for them, so an executor hands back whatever it does
    not send elsewhere. ``record_progress`` counts items it finished elsewhere.
    ``portable_item`` says whether another process can run an item: the
    resolver another process rebuilds may know fewer kinds of item than the
    run's own (see ``task_resolvers``). ``source_fields`` gives what a result
    entry says about where an item came from (its name, location and label),
    for the entries of items that ran elsewhere.
    """

    run_in_process: Callable[[list[Any]], Awaitable[list[Any]]]
    record_progress: Callable[[int], Awaitable[None]]
    portable_item: Callable[[Any], bool]
    source_fields: Callable[[Any], dict[str, Any]]


class DataItemExecutor(Protocol):
    """Runs the data items of a run ``run_tasks`` owns.

    Only a run that may leave its process reaches an executor (see
    ``portable_spec``): an incremental one, among other things.

    Returns one entry per item, in any order. For an item it handed back, the
    entry ``item_run.run_in_process`` returned. For an item that ran elsewhere,
    a dict with ``run_info`` (a ``PipelineRunCompleted``,
    ``PipelineRunAlreadyCompleted`` or ``PipelineRunErrored``), ``data_id`` (the
    item's id), what ``item_run.source_fields`` says of the item and, for a
    failed item, ``error`` (an exception named after the failure):
    ``ItemOutcome.to_entry`` builds it.

    The run's rollback reads the entries: it keeps an item this run completed
    and removes the work of the others. When ``run_items`` raises instead, an
    item handed back with ``run_in_process`` raising included, the rollback gets
    no entries and goes by each item's completion stamp alone, as startup
    recovery does. Either way nothing a process finished is undone.

    When ``run_items`` returns or raises, cancellation included, nothing it
    started may still be running: ``run_tasks`` rolls a failed run back and
    releases the run's ownership right after.
    """

    async def run_items(
        self, spec: PipelineRunSpec, data_items: list[Any], item_run: ItemRun
    ) -> list[Any]: ...


_executor: DataItemExecutor | None = None


def set_data_item_executor(executor: DataItemExecutor | None) -> None:
    """Register the executor every portable run in this process goes through."""
    global _executor
    _executor = executor


def get_data_item_executor() -> DataItemExecutor | None:
    return _executor


def portable_spec(
    run: RunSettings,
    task_resolver: Any,
    dataset_id: UUID,
    user_id: UUID,
    llm_config: Any,
    embedding_config: Any,
) -> PipelineRunSpec | None:
    """``run`` as another process needs it described, or ``None`` when its data
    items must run in the process that owns it. They may leave it only when:

    - the run's resolver is portable, so another process can rebuild it;
    - the run has no per-call LLM or embedding configuration, its own or one an
      outer scope set for this context: those objects hold credentials that do
      not travel;
    - the run's vector and graph stores are the ones another process would open
      too. With backend access control every process binds a dataset's own
      stores; without it, a store configuration a call set for this context is
      known to this process alone;
    - the run is incremental. Only then does a completion stamp mean "this item
      is done", which is all the rollback has to go by when the executor fails
      without results. A run that is not incremental reprocesses stamped items
      and stamps none;
    - what it would send survives JSON.
    """
    portable = getattr(task_resolver, "portable", None)
    if (
        portable is None
        or llm_config is not None
        or embedding_config is not None
        or context_llm_config.get() is not None
        or context_embedding_config.get() is not None
        or not (run.incremental_loading or run.data_cache)
        or (
            not backend_access_control_enabled()
            and (
                context_vector_db_config.get() is not None
                or context_graph_db_config.get() is not None
            )
        )
    ):
        return None
    resolver_name, resolver_args = portable
    spec = PipelineRunSpec(
        **vars(run),
        dataset_id=dataset_id,
        user_id=user_id,
        task_resolver=resolver_name,
        task_resolver_args=resolver_args,
    )
    return spec if spec.survives_json() else None


async def run_items_with(
    executor: DataItemExecutor, spec: PipelineRunSpec, data_items: list[Any], item_run: ItemRun
) -> list[Any]:
    """Have ``executor`` run ``data_items``, and hold its answer to what ``run_tasks`` relies on.

    Every item must be answered for exactly once: ``run_tasks`` completes a run
    whose entries hold no failure, so an executor that answers for fewer items,
    or for other items than it was handed, would leave items unprocessed under a
    completed run, and one that answers twice for an item would count it twice.

    An item that ran elsewhere cannot raise here; its failure arrives recorded.
    A budget rejection is the failure callers map to a status of its own (402),
    so it is raised as the typed error an item run in this process lets through.
    """
    entries = await executor.run_items(spec, data_items, item_run)

    answered = [entry for entry in entries if isinstance(entry, dict)]
    data_ids = [entry["data_id"] for entry in answered if entry.get("data_id") is not None]
    # A data row is answered for under its own id. (Other kinds of item get
    # their id while they run, so for those the count is all there is to check.)
    unanswered = {item.id for item in data_items if isinstance(item, Data)} - set(data_ids)
    if len(entries) != len(data_items) or len(data_ids) != len(set(data_ids)) or unanswered:
        raise RuntimeError(
            f"The data item executor was handed {len(data_items)} items and returned "
            f"{len(entries)} entries ({len(data_ids) - len(set(data_ids))} of them for an item "
            f"already answered for), with none for {len(unanswered)} of the items. Every item "
            "must be answered for exactly once."
        )
    for entry in answered:
        run_info = entry.get("run_info")
        if isinstance(run_info, PipelineRunErrored):
            raise_if_budget_exhausted_record(run_info.error_class, run_info.error_message)
    return entries


def split_completed(
    data_items: list[Any], pipeline_name: str, dataset_id: UUID
) -> tuple[list[Any], list[Any]]:
    """``(already completed, still to process)`` for this pipeline and dataset.

    The same stamp ``run_tasks_data_item`` skips an item on, so an executor
    sizes its work by what will actually run.
    """
    completed, pending = [], []
    for item in data_items:
        (completed if is_item_stamped(item, pipeline_name, dataset_id) else pending).append(item)
    return completed, pending
