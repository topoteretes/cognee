"""Functions plugins run on chunks right after chunking (CLO-397).

A plugin that must see document chunks before anything indexes, extracts or
stores them (a gate that screens chunk text, a tagger) registers a function
here::

    async def screen_chunks(chunks, ctx):
        return [chunk for chunk in chunks if allowed(chunk)]

    register_chunk_task("memory_gate", screen_chunks)

It receives a batch of chunks and the run's ``PipelineContext`` (user, dataset,
``pipeline_name``) and returns the chunks to keep. Registered functions run in
registration order, wherever cognee has fresh chunks: in every task list that
chunks documents (cognify's LLM list, its GLiNER demo list, the indexing
pipeline's list and the dlt route's list, whose chunks are source rows), as one
task right after the chunking task, and in ``update()``'s chunk-level path, on
the chunks an edit produced. With nothing registered the lists are unchanged.

* Return the chunks to keep, also when the function only changes them in
  place: returning ``None`` raises.
* The registry is process-wide, so every dataset's chunks pass through: a
  choice per dataset is made in the function, from ``ctx``.
* ``needs_llm=True`` when the function calls the LLM, so cognify checks the LLM
  connection before the run. The default keeps an install with no LLM key (the
  GLiNER route) LLM-free.
* cognify runs the indexing pipeline (unless ``EARLY_CHUNK_INDEXING`` is off)
  and then the cognify pipeline over the same chunks, and both run the
  functions. Give a chunk the same answer in both runs: a chunk the indexing
  run stored and the cognify run dropped stays searchable, without its entities.
* A document that lost a chunk no longer matches its stored text, so the next
  ``update()`` of it rebuilds the whole document (``chunks_not_tiling``).

Code files and code repositories are not chunked and run none of this, and
neither does a pipeline that builds chunks without chunking a document (the
code docs memify pipeline).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cognee.pipelines.types import Drop

if TYPE_CHECKING:
    from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
    from cognee.modules.pipelines.models import PipelineContext

ChunkTask = Callable[["list[DocumentChunk]", "PipelineContext"], Awaitable["list[DocumentChunk]"]]


@dataclass(frozen=True)
class RegisteredChunkTask:
    run: ChunkTask
    needs_llm: bool


_chunk_tasks: dict[str, RegisteredChunkTask] = {}


def register_chunk_task(name: str, task: ChunkTask, needs_llm: bool = False) -> None:
    """Run ``task`` on every batch of fresh chunks, right after chunking.

    Registering a name again replaces its function and keeps its place;
    ``unregister_chunk_task`` removes it.
    """
    if not callable(task):
        raise TypeError(f"register_chunk_task() takes an async function, got {type(task).__name__}")
    _chunk_tasks[name] = RegisteredChunkTask(task, needs_llm)


def unregister_chunk_task(name: str) -> ChunkTask | None:
    """Remove the function registered under ``name``; return it, or None if there was none."""
    registered = _chunk_tasks.pop(name, None)
    return registered.run if registered else None


def registered_chunk_tasks() -> list[RegisteredChunkTask]:
    """The registered functions in registration order, as a snapshot."""
    return list(_chunk_tasks.values())


async def apply_chunk_tasks(chunks: list, ctx) -> list:
    """The chunks the registered functions keep, in registration order."""
    for name, task in list(_chunk_tasks.items()):
        if not chunks:
            break
        chunks = await task.run(chunks, ctx)
        if chunks is None:
            raise TypeError(f"Chunk task {name!r} returned None; return the chunks to keep.")
    return chunks


async def run_chunk_tasks(chunks: list, ctx=None):
    """``apply_chunk_tasks`` as a pipeline task: a batch nothing is left of is dropped."""
    return await apply_chunk_tasks(chunks, ctx) or Drop
