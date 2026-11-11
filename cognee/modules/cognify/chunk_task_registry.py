"""Tasks plugins add right after chunking (CLO-397).

A plugin that must see document chunks before anything indexes, extracts or
stores them (a gate that screens chunk text, a tagger) registers a factory
here. Every task list that chunks documents runs the tasks the factories
return right after its chunking task, in registration order: cognify's LLM
list, its GLiNER demo list and the indexing pipeline's list, which build that
step with ``build_chunking_tasks`` (``chunking_tasks.py``), and the dlt
route's list, where each row of a relational source arrives as one
``DltRow`` chunk (a ``DocumentChunk`` subclass) and no text is cut. With
nothing registered the lists are unchanged.

Writing the task (runner semantics: ``cognee/modules/pipelines/__init__.py``):

* It receives a list of chunks and returns the chunks the next task gets.
  A task's ``batch_size`` batches the *previous* task's output, so yours sets
  how many chunks the chunker hands it per call, and a coroutine's return
  value reaches the next task as one batch: set it to the ``chunks_per_batch``
  the factory receives, or the tasks after it run on smaller batches.
* ``enriches=True`` when it only changes chunks in place (return ``None`` and
  the input passes on). To drop chunks, return the ones to keep, and ``Drop``
  (``cognee.pipelines``) when none are left: extraction refuses an empty list.
* ``needs_llm=False`` unless it calls the LLM. The default is True, and one
  such task makes cognify check the LLM connection before it runs, which
  fails an install with no LLM key (the GLiNER route).
* A ``ctx`` parameter receives the ``PipelineContext`` (user, dataset,
  ``pipeline_name``). The registry is process-wide, so every dataset's lists
  carry the task: a choice per dataset is made in the task, from ``ctx``.
* cognify runs the indexing pipeline (unless ``EARLY_CHUNK_INDEXING`` is off)
  and then the cognify pipeline over the same chunks, and both run the task.
  Give a chunk the same answer in both runs: a chunk the indexing run stored
  and the cognify run dropped stays searchable, without its entities.

Code files and code repositories are not chunked and run none of this, and
neither does a pipeline that builds chunks without chunking a document (the
code docs memify pipeline). ``update()``'s chunk-level path runs no task list,
so ``update()`` rebuilds the document through ``cognify()`` while the
factories return tasks: return an empty list while the plugin is switched off
to keep that path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognee.modules.pipelines.tasks.task import Task

# ``def factory(*, chunks_per_batch: int, **kwargs) -> Task | list[Task]``
ChunkTaskFactory = Callable[..., "Task | list[Task]"]

_chunk_task_factories: dict[str, ChunkTaskFactory] = {}


def register_chunk_task(name: str, factory: ChunkTaskFactory) -> None:
    """Run the tasks ``factory`` builds on the chunks, right after chunking.

    ``factory`` is called with keyword arguments each time cognee builds a
    task list that chunks documents (and by ``update()``, to learn whether any
    task would run), and returns a ``Task`` or a list of them (empty to add
    nothing). Accept ``**kwargs``: more may follow. Passed today:

    * ``chunks_per_batch``: the batch size of the list's later tasks; give it
      to your task as ``batch_size``.

    Tasks run in registration order. Registering a name again replaces its
    factory and keeps its place; ``unregister_chunk_task`` removes it::

        def gate_tasks(*, chunks_per_batch, **kwargs):
            return Task(screen_chunks, batch_size=chunks_per_batch, needs_llm=False)

        register_chunk_task("memory_gate", gate_tasks)
    """
    if not callable(factory):
        raise TypeError(
            f"register_chunk_task() takes a callable factory, got {type(factory).__name__}"
        )
    _chunk_task_factories[name] = factory


def unregister_chunk_task(name: str) -> ChunkTaskFactory | None:
    """Remove the factory registered under ``name``; return it, or None if there was none."""
    return _chunk_task_factories.pop(name, None)


def chunk_task_factories() -> list[tuple[str, ChunkTaskFactory]]:
    """The registered ``(name, factory)`` pairs in registration order, as a snapshot."""
    return list(_chunk_task_factories.items())
