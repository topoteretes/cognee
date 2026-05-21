"""
Pipeline API Proposal — What the simplified API would look like
================================================================

This is NOT runnable code. It's a design document showing how the three
PRs from the review plan would work together from a user's perspective.

Plan:
  PR A — Fix Task.__init__ to accept batch_size= directly (already done)
  PR B — Add Drop, semaphore concurrency, and enriches to run_tasks_base
  PR C — FieldAnnotations (Embeddable, Dedup, LLMContext) standalone

The key insight: we don't need a parallel execution engine (flow.py).
We improve the one we have.
"""

# =============================================================================
# 1. DEFINING DATA MODELS (PR C — FieldAnnotations)
# =============================================================================
#
# Before (current):
#
#   class Entity(DataPoint):
#       name: str
#       description: str
#       metadata: dict = {
#           "index_fields": ["name"],          # implicit: "this gets embedded"
#           "identity_fields": ["name"],        # implicit: "this deduplicates"
#       }
#
# After (with FieldAnnotations):

from typing import Annotated, Optional
from cognee.infrastructure.engine.models import DataPoint
from cognee.infrastructure.engine.models.FieldAnnotations import Embeddable, Dedup, LLMContext


class Entity(DataPoint):
    """Now the field roles are visible at definition time."""

    name: Annotated[str, Embeddable("Primary search field"), Dedup()]
    description: Annotated[str, LLMContext("Provides entity context to LLM")]
    is_a: Optional[str] = None

    # metadata is auto-generated from annotations:
    #   index_fields = ["name"]       (from Embeddable)
    #   identity_fields = ["name"]    (from Dedup)
    # No more manually maintaining metadata dicts.


class DocumentChunk(DataPoint):
    text: Annotated[str, Embeddable("Chunk text for semantic search")]
    document_id: str
    chunk_index: int


# =============================================================================
# 2. DEFINING PIPELINE TASKS (PR A — Task improvements, already landed)
# =============================================================================
#
# Before (old API):
#
#   Task(extract_graph, task_config={"batch_size": 10}, graph_model=KnowledgeGraph)
#
# After (current, already in this branch):

from cognee.modules.pipelines.tasks.task import Task, task  # noqa: E402
from cognee.pipelines.types import Drop  # noqa: E402


# Option A: Plain function + Task wrapper at pipeline definition
async def classify_documents(data):
    """Classify input documents by type."""
    # ... classification logic ...
    return classified_doc  # noqa: F821


async def extract_chunks(document):
    """Extract text chunks from a document. Yields chunks one at a time."""
    for chunk in document.split_into_chunks():
        yield DocumentChunk(text=chunk.text, document_id=document.id, chunk_index=chunk.idx)


async def extract_entities(chunks, graph_model=None):
    """Extract entities from chunks using LLM."""
    # ... LLM extraction logic ...
    return entities  # noqa: F821


async def filter_low_quality(entity):
    """Drop entities below quality threshold."""
    if entity.confidence < 0.5:
        return Drop  # <-- Item is removed from the pipeline
    return entity


async def add_to_graph(entities):
    """Store entities in graph + vector databases."""
    # ... storage logic ...
    return entities


# Option B: @task decorator (attaches .task attribute for convenience)
@task(batch_size=20)
async def extract_entities_decorated(chunks, graph_model=None):
    """Same function, but config is attached at definition time."""
    return []


# =============================================================================
# 3. BUILDING AND RUNNING PIPELINES (PR A + PR B)
# =============================================================================

# --- Simple pipeline: same as today, but Task() is cleaner ---


async def run_simple_pipeline(datasets, user):
    from cognee.modules.pipelines.operations.run_tasks import run_tasks

    tasks = [
        Task(classify_documents),
        Task(extract_chunks, batch_size=5),  # was: task_config={"batch_size": 5}
        Task(extract_entities, batch_size=20, graph_model=KnowledgeGraph),  # noqa: F821
        Task(filter_low_quality),  # uses Drop to remove items
        Task(
            add_to_graph, batch_size=50, enriches=True
        ),  # enriches=True: returns input if fn returns None
    ]

    # run_tasks already handles: batching, error continuation, observability,
    # telemetry, provenance stamping. No need to reimplement.
    pipeline = run_tasks(tasks, datasets=datasets, user=user)

    async for status in pipeline:
        print(status)


# --- Using @task decorator + .task attribute ---


async def run_decorated_pipeline(datasets, user):
    from cognee.modules.pipelines.operations.run_tasks import run_tasks

    tasks = [
        Task(classify_documents),
        extract_entities_decorated.task,  # uses config from @task(batch_size=20)
        extract_entities_decorated.task.with_config(batch_size=10),  # override at call site
    ]

    pipeline = run_tasks(tasks, datasets=datasets, user=user)
    async for status in pipeline:
        print(status)


# =============================================================================
# 4. WHAT PR B ADDS TO run_tasks_base (NOT a new execution engine)
# =============================================================================
#
# PR B brings three improvements INTO the existing run_tasks_base:
#
# 4a. Semaphore-based concurrency (replaces batch-based parallelism)
# ─────────────────────────────────────────────────────────────────
#
#   Before: run_tasks_parallel splits into fixed batches, runs each batch fully
#           before starting the next. Wasteful if items have variable latency.
#
#   After:  A semaphore limits concurrent items. Fast items free slots for others.
#           This is a change INSIDE run_tasks.py, not a new file.
#
#   Usage (from the user's perspective, nothing changes):
#
#     pipeline = run_tasks(tasks, datasets=datasets, user=user, max_parallel=20)
#
#
# 4b. Drop sentinel (already in Task — just needs to be documented)
# ─────────────────────────────────────────────────────────────────
#
#   Return Drop from any task to remove that item from the pipeline.
#   Already implemented in Task.execute_coroutine / execute_function.
#   No new code needed — just documentation and tests.
#
#
# 4c. enriches=True (already in Task — same story)
# ─────────────────────────────────────────────────
#
#   When enriches=True, if the task returns None, the original input
#   is passed through unchanged. Already implemented. Just needs docs.


# =============================================================================
# 5. WHAT WE DELETE (things from the current PR that are unnecessary)
# =============================================================================
#
# These files are replaced by improvements to the existing infrastructure:
#
#   DELETE cognee/pipelines/flow.py      → semaphore goes into run_tasks.py
#   DELETE cognee/pipelines/step.py      → @task decorator already exists in task.py
#   DELETE cognee/pipelines/builder.py   → Pipeline builder adds complexity without value
#   DELETE cognee/pipelines/context.py   → cognee_pipeline() duplicates existing setup
#   DELETE cognee/pipelines/types.py     → Drop already in task.py, Pipe[T]/Ctx[T] unused
#
# KEEP:
#   cognee/infrastructure/engine/models/FieldAnnotations.py  → genuinely useful
#   cognee/pipelines/__init__.py  → legacy compat re-exports (simplified)


# =============================================================================
# 6. COMPLETE WORKING EXAMPLE — What a user would actually write
# =============================================================================


async def full_example():
    """End-to-end example with the proposed API."""
    import cognee
    from cognee.low_level import setup
    from cognee.modules.data.methods import load_or_create_datasets
    from cognee.modules.users.methods import get_default_user
    from cognee.modules.pipelines.operations.run_tasks import run_tasks
    from cognee.modules.pipelines.tasks.task import Task, task
    from cognee.pipelines.types import Drop
    from cognee.tasks.storage import add_data_points

    # --- Define models with explicit field roles ---

    class Person(DataPoint):
        name: Annotated[str, Embeddable(), Dedup()]
        age: int = 0

    class Department(DataPoint):
        name: Annotated[str, Embeddable(), Dedup()]
        employees: list[Person] = []

    # --- Define tasks (plain functions) ---

    async def parse_people(raw_data):
        """Parse raw JSON into Person DataPoints."""
        for person in raw_data["people"]:
            yield Person(name=person["name"], age=person.get("age", 0))

    async def filter_adults(person):
        """Only keep adults. Drop minors."""
        if person.age < 18:
            return Drop
        return person

    async def group_by_department(person):
        """Enrich: tag person with department."""
        # enriches=True means: if we return None, pass person through unchanged
        person.department_tag = "engineering"  # simplified
        return person

    # --- Run pipeline ---

    await cognee.prune.prune_data()
    await setup()

    user = await get_default_user()
    datasets = await load_or_create_datasets(["example"], [], user)

    raw_data = {
        "people": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 16},
            {"name": "Charlie", "age": 25},
        ]
    }

    tasks = [
        Task(parse_people),
        Task(filter_adults),  # Bob gets Dropped
        Task(group_by_department, enriches=True),
        Task(add_data_points, batch_size=50),  # store in graph + vector
    ]

    pipeline = run_tasks(
        tasks,
        datasets=[datasets[0].id],
        data=[raw_data],
        user=user,
        pipeline_name="example_pipeline",
    )

    async for status in pipeline:
        print(status)

    # Result: Alice and Charlie stored with deterministic UUIDs (via Dedup),
    # embedded in vector DB (via Embeddable), Bob filtered out (via Drop).


# =============================================================================
# 7. COMPARISON TABLE
# =============================================================================
#
# | Feature                  | Current PR (flow.py)     | Proposed (improve existing) |
# |--------------------------|--------------------------|----------------------------|
# | Execution engine         | New 439-line reimpl      | Enhance run_tasks_base     |
# | Observability/telemetry  | Missing                  | Already there              |
# | Provenance stamping      | Missing                  | Already there              |
# | Drop sentinel            | Reimplemented            | Already in Task            |
# | enriches=True            | Reimplemented            | Already in Task            |
# | batch_size config        | @step decorator          | Task(fn, batch_size=N)     |
# | Semaphore concurrency    | In flow.py               | Move into run_tasks.py     |
# | Type validation          | _is_obvious_mismatch     | Remove (too weak to trust) |
# | Context injection        | Ctx[T] (not implemented) | Explicit function args     |
# | Field annotations        | FieldAnnotations.py      | Keep as-is (PR C)          |
# | Lines of new code        | ~1200                    | ~100                       |
# | Files added              | 6                        | 0 (modify 2 existing)      |
# | Maintenance surface      | 2 parallel engines       | 1 engine                   |
