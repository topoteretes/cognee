---
name: cognee-custom-pipelines
description: Use when building your own cognee processing — writing custom tasks, chaining them into a pipeline with run_custom_pipeline or cognee.pipelines.run_pipeline, storing custom DataPoints with add_data_points, running custom extraction/enrichment over the existing graph with memify, checking pipeline run status, or debugging how data flows between tasks (batch_size, data_per_batch, ctx, Drop, enriches).
---

# Custom tasks and pipelines

Everything cognee does runs as a **pipeline**: an ordered list of **tasks**,
each a plain Python function whose output feeds the next one. `remember()`
is the right tool for ordinary ingestion. Build a pipeline when you need
processing cognee does not ship: your own extraction, your own node types,
or a post-processing step over the graph.

```python
import cognee
from cognee.modules.pipelines import Task
from cognee.tasks.storage import add_data_points
from cognee.low_level import DataPoint

class Person(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}

async def extract_people(data_items: list) -> list[Person]:
    people = []
    for item in data_items:                       # always a list, see below
        text = item if isinstance(item, str) else ""
        people += [Person(name=n.strip()) for n in text.split(",") if n.strip()]
    return people

result = await cognee.run_custom_pipeline(
    tasks=[
        Task(extract_people, needs_llm=False),
        Task(add_data_points, needs_llm=False),    # store in graph + vector DBs
    ],
    data=["Ada Lovelace, Alan Turing"],
    dataset="people",
)
```

## Use it

### Pick the runner

There are three, and two share the name `run_pipeline`:

| Runner | Import | Use it for |
|---|---|---|
| `cognee.run_custom_pipeline(...)` | `cognee` | The normal choice: runs your tasks against a dataset with permissions, a per-dataset lock, run records, and status |
| Full orchestrator `run_pipeline(tasks=..., data=..., datasets=...)` | `cognee.modules.pipelines` | What `run_custom_pipeline` and `cognify` call; yields `PipelineRunInfo` |
| Lightweight `run_pipeline([...], data=...)` | `cognee.pipelines` | Quick chains of `task()` specs with no permissions, locks, run rows, or migrations; returns the last step's outputs |

`cognee.run_custom_pipeline(tasks, data=None, dataset="main_dataset",
user=None, incremental_loading=False, data_per_batch=20,
run_in_background=False, pipeline_name="custom_pipeline", data_cache=False,
...)` returns `{dataset_id: PipelineRunInfo}` (the started run when
`run_in_background=True`). With `data=None` it runs over the dataset's
existing documents (`Data` rows).

### Write a task

```python
from cognee.modules.pipelines import Task
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import task_summary
from cognee.pipelines import Drop

@task_summary("Tagged {n} chunk(s)")
async def tag_chunks(chunks: list, ctx: PipelineContext = None, label: str = "x"):
    for chunk in chunks:
        chunk.metadata["label"] = label
    return chunks            # or yield per item; return/yield Drop to discard

tag = Task(tag_chunks, label="reviewed", batch_size=10, needs_llm=False)
```

- A task is an `async def`, a generator, an async generator, or a plain
  `def`. Extra `Task(fn, *args, **kwargs)` arguments are passed after the
  pipeline data.
- `needs_llm=False` on tasks that never call an LLM lets an LLM-free
  pipeline skip the LLM connection check.
- `ctx` (injected by the parameter **name** `ctx`) carries `user`,
  `data_item`, `dataset`, `pipeline_run_id`, `pipeline_name`, and `extras`.
- `task.with_config(batch_size=..., **kwargs)` returns a modified copy.

### How data flows

- **Each document runs the whole chain on its own**, and the first task
  receives it as a **one-element list** (`[data_item]`), not the bare item.
- **`data_per_batch`** (default 20) is how many documents run at the same
  time. It is a concurrency limit, not a batch size.
- **`batch_size` belongs to the consumer.** A task's `batch_size` decides how
  the *previous* task's generator output is grouped before it is passed in.
  Generator tasks always hand over lists; a coroutine or function hands over
  its single return value.
- **Streaming:** each upstream result goes down the chain immediately, so a
  downstream task can run many times per document.
- **`enriches=True`:** if the task returns `None`, its input is passed on
  unchanged (coroutines and functions only, not generators).
- **`Drop`:** returning or yielding it removes that item from the stream.
- Every `DataPoint` passing through is stamped automatically with where it
  came from (`source_pipeline`, `source_task`, `source_user`, …).

### Store results

`add_data_points(data_points, custom_edges=None, embed_triplets=False,
graph_only=False)` writes a list of `DataPoint`s to the graph and indexes
their `index_fields` in the vector DB. It returns the same list, so it can
sit mid-chain. Give every node type `identity_fields` so repeated runs merge
instead of duplicating (see the `cognee-custom-graph-models` skill).

### Work on the existing graph: memify

```python
await cognee.memify(
    extraction_tasks=["extract_subgraph_chunks"],        # names or Task objects
    enrichment_tasks=[Task(my_enrichment, needs_llm=False)],
    dataset="people",
    node_name=["AI"],                                     # optional subgraph filter
)
```

With no `data`, memify passes the graph (or the `node_type` / `node_name`
subgraph) to the first task. Registered task names:
`extract_subgraph`, `extract_subgraph_chunks`, `get_triplet_datapoints`,
`extract_user_sessions`, `cognify_session`, `extract_agent_trace_feedbacks`,
`cognify_agent_trace_feedback`, `apply_feedback_weights`,
`detect_entity_duplicates`, `merge_entity_duplicates`, `index_data_points`.
`improve()` also forwards `extraction_tasks` / `enrichment_tasks` to memify,
but only inside its enrichment stage, which can be skipped when nothing
changed. Call `memify` directly when you want your tasks to run every time.

### Check status

```python
status = await cognee.datasets.get_status([dataset_id], pipeline_names=["custom_pipeline"])
```

Without `pipeline_names` it reports only `cognify_pipeline`. It returns
`{dataset_id: PipelineRunStatus}`: `DATASET_PROCESSING_INITIATED`,
`_STARTED`, `_COMPLETED`, or `_ERRORED`. The value `run_custom_pipeline`
returns per dataset is a `PipelineRunInfo` instead, whose class names the
outcome: `PipelineRunCompleted`, `PipelineRunAlreadyCompleted`,
`PipelineRunErrored`, and so on.

## Pitfalls

- **Wrong `run_pipeline`.** The one in `cognee.pipelines` wants `task()`
  specs *called* (`extract()`, not `extract`) and raises `TypeError`
  otherwise; the orchestrator in `cognee.modules.pipelines` wants `Task`
  objects and raises `WrongTaskTypeError` otherwise.
- **String task names only work in `memify`.** `run_custom_pipeline` accepts
  only `Task` objects despite its type hint.
- **Some callables are rejected** by `Task` (`ValueError: Unsupported task
  type`): *sync* bound methods, `functools.partial` of a sync function, and
  callable objects (instances with `__call__`). Async methods, partials of
  async functions, plain functions, and lambdas work. When in doubt, wrap it
  in a plain `def` / `async def`.
- **`run_custom_pipeline` does not run database migrations.** On an existing
  database, run `await cognee.run_migrations()` (or any `remember()` first).
- **Keep `pipeline_name="custom_pipeline"`** unless you add your name to
  `WRITE_PIPELINE_NAMES` in `cognee/modules/improve/graph_changes.py`.
  Otherwise `improve()` does not notice your graph writes and may skip
  enrichment as "already completed".
- **memify defaults.** An omitted or empty task list is replaced by the
  defaults (`get_triplet_datapoints` extraction, `index_data_points`
  enrichment), and memify uses only the first dataset it resolves.
- **Nodes duplicate on every run** when a DataPoint has no
  `identity_fields`. Several shipped examples have this bug; don't copy it.

## How it works

`run_custom_pipeline` → orchestrator `run_pipeline` (checks write
permission, takes the per-dataset lock, records a `PipelineRun`) →
`run_tasks` (a semaphore of `data_per_batch`, one chain per document) →
`run_tasks_base` (streams each task's output into the next, batching by the
consumer's `batch_size`, injecting `ctx`, stamping provenance).

- Package overview and the runner semantics: `cognee/modules/pipelines/__init__.py`
- `Task`, `task()`, `TaskSpec`, `BoundTask`, `@task_summary`: `cognee/modules/pipelines/tasks/task.py`
- Orchestrator: `cognee/modules/pipelines/operations/pipeline.py`; execution:
  `run_tasks.py`, `run_tasks_base.py`, `run_tasks_data_item.py`
- Lightweight runner: `cognee/modules/pipelines/operations/run_pipeline.py`,
  exported from `cognee/pipelines/`
- Context: `cognee/modules/pipelines/models/PipelineContext.py`
- `run_custom_pipeline`: `cognee/modules/run_custom_pipeline/run_custom_pipeline.py`
- memify: `cognee/modules/memify/memify.py`,
  `cognee/memify_pipelines/memify_task_registry.py`, `memify_default_tasks.py`
- Storage: `cognee/tasks/storage/add_data_points.py`
- Index of all shipped tasks: `cognee/tasks/README.md`

Examples:

- `examples/demos/custom_pipelines/custom_pipeline_single_object_example.py`:
  the best reference. It runs over added documents, does LLM extraction into
  typed DataPoints, then recalls. Add `identity_fields` to its models.
- `examples/demos/custom_pipelines/organizational_hierarchy/`: low-level
  `run_tasks`, no LLM, dedup via `identity_fields`, status polling.
- `examples/demos/custom_pipelines/custom_cognify_pipeline_example.py`:
  rebuilds add + cognify from the default task list.
- `examples/demos/custom_pipelines/memify_coding_agent_rule_extraction_example.py`:
  memify with a custom enrichment task.

## Extending it

- **A new shipped task:** put it in the `cognee/tasks/` subpackage for its
  stage, export it from that package's `__init__.py`, follow the template in
  `cognee/tasks/README.md`, and add a unit test under `cognee/tests/unit/tasks/`.
- **A new memify task name:** register it in
  `cognee/memify_pipelines/memify_task_registry.py`.
- **A new write pipeline name:** add it to `WRITE_PIPELINE_NAMES`.
- Pipeline tests: `cognee/tests/unit/modules/pipelines/` (runner semantics,
  context, provenance, rollback) and `cognee/tests/unit/pipelines/` (the
  lightweight API).
