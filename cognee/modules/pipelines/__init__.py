"""Pipeline engine: how cognee runs a list of tasks over data.

Every ingestion or enrichment flow (``cognify``, ``improve``/``memify``, the code
graph, custom pipelines) is a list of tasks executed by the runner in this
package. Task implementations live in ``cognee/tasks/``; this package only
knows how to run them.

Four things named "task" -- do not confuse them
------------------------------------------------
* ``Task`` (``tasks/task.py``) -- the runtime unit. Wraps any callable: plain
  function, coroutine, generator or async generator. ``Task(fn, *args,
  batch_size=..., enriches=..., needs_llm=..., **kwargs)``. This is what
  ``cognee/tasks/*`` produce and what ``run_tasks``/``run_pipeline`` execute.
* ``TaskSpec`` / ``BoundTask`` / ``task()`` (same module) -- the deferred-call
  sugar: ``spec = task(fn, batch_size=20)``; ``spec(graph_model=KG)`` returns a
  ``BoundTask`` with kwargs captured for later. Used with the *new*
  ``run_pipeline`` below.
* ``models/Task.py`` -- a SQLAlchemy table (``tasks``) that records pipeline
  definitions in the relational DB. Nothing to do with execution.
* ``@task_summary("Did {n} thing(s)")`` -- attaches a human-readable template
  used for telemetry/result summaries.

Two functions named ``run_pipeline`` -- pick the right one
----------------------------------------------------------
* ``cognee.modules.pipelines.run_pipeline`` (``operations/pipeline.py``) -- the
  full orchestrator: resolves the user and authorized datasets, acquires the
  per-dataset lock and database context, records ``PipelineRun`` rows, and
  yields ``PipelineRunInfo`` events. Takes ``tasks: list[Task]`` (or a
  per-item task resolver), ``data``, ``datasets``, ``user``,
  ``pipeline_name``. This is what ``cognify``, ``improve`` and
  ``run_custom_pipeline`` call. Use it when the run must be dataset-aware,
  permission-checked or resumable.
* ``cognee.pipelines.run_pipeline`` (``operations/run_pipeline.py``) -- the
  lightweight deferred-call runner: ``run_pipeline([spec1(), spec2(...)],
  data=...)`` over ``BoundTask`` steps, returns the final step's results as a
  list. No dataset locking, no run records. Use it for in-process
  composition and tests.

Execution model
---------------
``run_tasks_base`` (``operations/run_tasks_base.py``) is recursive and
streaming, not a map over a list:

1. Task 0 runs with the pipeline input as its first positional argument.
2. Every result it emits is passed to the *rest of the chain* immediately
   (``run_tasks_base(leftover_tasks, result, ...)``), so downstream tasks may
   run many times per pipeline -- once per upstream batch.
3. **A task's ``batch_size`` batches the *previous* task's output.** The runner
   reads ``next_task.task_config["batch_size"]`` and hands the current task
   that number so a generator task yields lists of that size. A coroutine
   task yields its single return value regardless of batch size.
4. ``Drop`` (``cognee.pipelines.Drop``) returned or yielded from a task removes
   that item from the stream.
5. ``enriches=True`` marks a task that mutates its input in place: when it
   returns ``None`` the runner passes the *input* through unchanged, so the
   next task still receives the data.
6. ``ctx: PipelineContext`` is injected into any task whose signature has a
   parameter literally named ``ctx`` (``Task.accepts_ctx``); it carries
   ``user``, ``dataset``, ``pipeline_run_id``, ``pipeline_name`` and a free
   ``extras`` dict.
7. ``needs_llm=False`` on every task lets the run skip the first-use LLM
   connection probe (``pipeline_needs_llm``).

Package layout
--------------
``operations/`` runners and run-lifecycle logging; ``layers/`` the checks a
run passes through (environment setup, dataset authorization, task
validation, qualification); ``methods/`` relational-DB reads and resets of
``PipelineRun`` rows; ``models/`` SQLAlchemy models plus ``PipelineContext``
and the ``PipelineRunInfo`` event types; ``queues/`` in-process progress
queues for background runs; ``utils/`` deterministic pipeline/run id
generation.
"""

from .tasks.task import Task
from .operations.run_tasks import run_tasks
from .operations.run_parallel import run_tasks_parallel
from .operations.pipeline import run_pipeline
