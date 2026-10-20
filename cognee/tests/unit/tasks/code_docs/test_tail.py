"""code_docs_tail: the chain in order, with the context, and the counts it reports."""

import importlib
import inspect
from types import SimpleNamespace

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.tasks.code_docs import CodeDocument, code_docs_tail
from cognee.tasks.code_docs.add_code_doc_edges import LinkCounts, SweepCounts
from cognee.tasks.code_docs.models import CodeDocsBatch
from cognee.tasks.code_docs.store_code_docs import StoreCounts

tail_module = importlib.import_module("cognee.tasks.code_docs.tail")

# The extraction's record: the repository it read, with the snapshot its graph holds.
READ_SNAPSHOTS = {"demo_repo": "snapshot-1"}


def _batch():
    document = CodeDocument(
        name="app/db.py",
        raw_data_location="app/db.py",
        external_metadata=None,
        repo="demo_repo",
        file_path="app/db.py",
        symbol_count=2,
    )
    chunks = [
        DocumentChunk(
            text=f"app/db.py:{index + 1}\napp/db.f{index} (function)\n\nDoc {index}.",
            chunk_size=5,
            chunk_index=index,
            cut_type="code_docstring",
            is_part_of=document,
        )
        for index in range(2)
    ]
    return CodeDocsBatch.of([document, *chunks], READ_SNAPSHOTS)


@pytest.fixture
def chain(monkeypatch):
    """The four steps as recorders: what each was called with, in order."""
    batch = _batch()
    steps = SimpleNamespace(
        batch=batch, cognified=CodeDocsBatch.carry(batch, ["summaries"]), calls=[]
    )

    def step(name, result):
        async def run(*args, **kwargs):
            steps.calls.append((name, args, kwargs))
            return result

        return run

    monkeypatch.setattr(tail_module, "extract_code_docs", step("extract", batch))
    monkeypatch.setattr(tail_module, "cognify_code_doc_chunks", step("cognify", steps.cognified))
    monkeypatch.setattr(
        tail_module, "store_new_code_docs", step("store", StoreCounts(documents=1, chunks=2))
    )
    monkeypatch.setattr(
        tail_module,
        "link_code_docs",
        step("link", LinkCounts(edges=2, swept=SweepCounts(chunks=1, owned=3))),
    )
    return steps


@pytest.mark.asyncio
async def test_embed_only_tail_runs_extract_store_link_with_the_context(chain):
    ctx = SimpleNamespace(dataset=SimpleNamespace(id="ds-1"))

    counts = await code_docs_tail("demo_repo", ctx)

    assert counts == {"documents": 1, "chunks": 2, "new": 2, "edges": 2, "swept": 4}
    assert chain.calls == [
        ("extract", (), {"repo": "demo_repo", "ctx": ctx, "snapshot_dir": None}),
        ("store", (chain.batch,), {"ctx": ctx, "graph_only": False}),
        ("link", (chain.batch,), {"ctx": ctx, "repo": "demo_repo"}),
    ]
    # The link task is handed the extraction's own batch, its record with it.
    assert chain.calls[2][1][0] is chain.batch


@pytest.mark.asyncio
async def test_tail_hands_the_snapshot_to_the_extraction_alone(chain):
    """The routes pass the snapshot they just loaded, and the extraction reads its
    docs.jsonl. The link task's sweep follows the batch's extraction record (which
    repositories were read, from which snapshot), not the directory; the store has
    no use for it either."""
    await code_docs_tail("demo_repo", snapshot_dir="/staging/demo_repo/.enola")

    assert chain.calls[0] == (
        "extract",
        (),
        {"repo": "demo_repo", "ctx": None, "snapshot_dir": "/staging/demo_repo/.enola"},
    )
    assert "snapshot_dir" not in chain.calls[1][2]
    assert chain.calls[2] == ("link", (chain.batch,), {"ctx": None, "repo": "demo_repo"})
    assert chain.calls[2][1][0].read_snapshots == READ_SNAPSHOTS


@pytest.mark.asyncio
async def test_entity_mode_tail_hands_the_cognified_batch_on(chain):
    counts = await code_docs_tail(None, extract_entities=True, graph_only=True)

    # The counts of documents and chunks are the extraction's, before the
    # chunks became summaries.
    assert counts["documents"] == 1 and counts["chunks"] == 2
    assert chain.calls == [
        ("extract", (), {"repo": None, "ctx": None, "snapshot_dir": None}),
        ("cognify", (chain.batch,), {"ctx": None}),
        ("store", (chain.cognified,), {"ctx": None, "graph_only": True}),
        ("link", (chain.cognified,), {"ctx": None, "repo": None}),
    ]
    # What cognify returned, the extraction's record carried over, reaches the link task.
    assert chain.calls[3][1][0] is chain.cognified


def test_tail_signature_is_the_shared_contract():
    parameters = inspect.signature(code_docs_tail).parameters
    assert list(parameters) == ["repo", "ctx", "extract_entities", "graph_only", "snapshot_dir"]
    assert parameters["ctx"].default is None
    assert parameters["extract_entities"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["extract_entities"].default is False
    assert parameters["snapshot_dir"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["snapshot_dir"].default is None
