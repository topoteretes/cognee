"""The one shape every maintenance job implements.

A maintenance job is upkeep a pipeline run leaves to do: reclaiming storage
its writes left behind, sweeping records it orphaned. It runs once after a
pipeline run completes, inside that run's dataset database context and dataset
lock (see ``runner.run_maintenance``).

A job is a ``name``, the ``pipelines`` whose completed runs trigger it, a
``gate`` and a ``run``. Two rules come with it:

* **Bounded by work, never by time.** A job runs inline at the end of the run
  that triggered it, so it must cap its own work -- a number of tasks, rows,
  files or versions per run -- and leave the rest for the next run. There is
  no timeout to fall back on: a job stopped by a clock leaves its work in an
  unknown state and does a different amount on a slow disk than a fast one.
* **Never raises.** The run it follows has already completed; nothing a job
  does may change that. The runner records an exception as ``errored``, but a
  job should map its own failures onto a ``JobResult`` where it can.

To add a job: subclass ``BaseMaintenanceJob``, add an instance to
``registry.DEFAULT_JOBS``, and make sure the pipelines it names pass
``after_run_completed=run_maintenance`` (today: cognify).
"""

from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from .result import JobResult


@dataclass(frozen=True)
class MaintenanceContext:
    """The completed run a maintenance pass follows."""

    pipeline_name: str
    pipeline_run_id: UUID | None
    dataset: Any
    user: Any
    # False for every dataset but the last of a multi-dataset run.
    last_in_invocation: bool = True


class BaseMaintenanceJob:
    """Default metadata and an open gate; concrete jobs override what they need."""

    name: str = ""
    # Pipeline names whose completed runs trigger this job.
    pipelines: frozenset[str] = frozenset()
    # What the job maintains. ``"dataset"``: one dataset's own data -- runs
    # after every dataset's run. ``"store"``: a whole database store -- with
    # multi-user off every dataset shares one store, so it runs once per
    # multi-dataset run, after the last dataset (with multi-user on, each
    # dataset has its own store and it runs after every dataset).
    scope: Literal["dataset", "store"] = "dataset"

    def gate(self, ctx: MaintenanceContext) -> str | None:
        """Return a skip reason, or ``None`` to run.

        Must be cheap: read configuration, never open a database or create an
        engine. A gate that raises skips the job (``gate_errored``).
        """
        return None

    async def run(self, ctx: MaintenanceContext) -> JobResult:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
