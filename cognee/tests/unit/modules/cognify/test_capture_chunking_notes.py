"""cognify's chunking manifest notes (SDK-529): the per-item task resolver records the
chunker and chunk size on the active run scope, read off the task list it resolved.

Two layers are covered. ``_note_chunking_config`` on its own, and the wiring: the
``tasks`` cognify() hands the pipeline executor is a per-item resolver, and it is that
resolver — invoked by ``run_tasks`` once per data item, INSIDE the pipeline's run
scope — which takes the note. The task lists themselves are built before any scope
exists, so a note taken at construction time would land nowhere; the wiring tests
therefore drive ``cognify()`` with a fake executor that resolves items inside a
``run_scope(kind="pipeline")`` exactly the way run_tasks does.
"""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.cognify.config import CognifyConfig
from cognee.modules.observability import capture
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents

# Module objects (not the re-exported cognify FUNCTION) for patch.object; dotted
# string targets break on Python 3.10 — see test_contradiction_detection_wiring.py.
cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
_mod_serve_state = importlib.import_module("cognee.api.v1.serve.state")
_mod_migrations_startup = importlib.import_module("cognee.modules.migrations.startup")

pytestmark = pytest.mark.usefixtures("capture_reset")


# ---------------------------------------------------------------------------
# The note helper on its own
# ---------------------------------------------------------------------------


def test_notes_chunker_and_chunk_size_from_the_resolved_task_list(fake_capture_sink):
    tasks = [
        Task(classify_documents),
        Task(extract_chunks_from_documents, max_chunk_size=321, chunker=TextChunker),
    ]

    with capture.run_scope(uuid4(), uuid4(), kind="pipeline") as scope:
        cognify_module._note_chunking_config(tasks)

    assert scope.fields["chunking.chunker"] == "TextChunker"
    assert scope.fields["chunking.chunk_size"] == 321


def test_a_task_list_without_a_chunking_task_notes_nothing(fake_capture_sink):
    with capture.run_scope(uuid4(), uuid4(), kind="pipeline") as scope:
        cognify_module._note_chunking_config([Task(classify_documents)])

    assert "chunking.chunker" not in scope.fields
    assert "chunking.chunk_size" not in scope.fields


def test_off_path_never_inspects_the_tasks(monkeypatch):
    monkeypatch.delenv("COGNEE_CAPTURE_ENABLED", raising=False)

    class _Explodes:
        @property
        def executable(self):
            raise AssertionError("tasks must not be inspected while capture is off")

    cognify_module._note_chunking_config([_Explodes()])

    assert capture.is_active() is False
    assert capture.current_scope() is None


# ---------------------------------------------------------------------------
# The wiring: cognify() -> resolver -> note, inside the pipeline run scope
# ---------------------------------------------------------------------------


def _text_item():
    return SimpleNamespace(system_metadata=None, extension="txt")


def _code_item():
    # Tagged at add time by ingest_data; routes to the LLM-free code list.
    return SimpleNamespace(system_metadata={"source": "code"}, extension="py")


async def _run_cognify_resolving(items, **cognify_kwargs):
    """Drive cognify() with a fake executor that resolves ``items`` inside a run scope.

    Mirrors run_tasks: the resolver cognify() passes as ``tasks`` is called once per
    data item inside ``capture.run_scope(..., kind="pipeline")``. Nothing else runs —
    the executor is where the pipeline would start. Returns the scope the resolver ran
    under and the task lists it resolved, in item order.
    """
    captured = {}

    def _fake_executor(run_in_background=False):
        async def _run(**executor_kwargs):
            resolver = executor_kwargs["tasks"]
            assert callable(resolver), "cognify() must hand the executor a per-item resolver"
            with capture.run_scope(uuid4(), uuid4(), kind="pipeline") as scope:
                captured["resolved"] = [resolver(item) for item in items]
            captured["scope"] = scope
            return {}

        return _run

    with (
        patch.dict(os.environ, {"TELEMETRY_DISABLED": "1"}),
        patch.object(cognify_module, "get_cognify_config", return_value=CognifyConfig()),
        patch.object(cognify_module, "get_pipeline_executor", _fake_executor),
        patch.object(_mod_migrations_startup, "run_migrations_and_block", new=AsyncMock()),
        patch.object(_mod_serve_state, "get_remote_client", return_value=None),
    ):
        await cognify_module.cognify(
            datasets=["ds"],
            # A non-None config skips the ontology-env branch; chunk_size is explicit
            # so get_max_chunk_tokens() (an LLM/embedding config read) never runs.
            config={"ontology_config": {"ontology_resolver": None}},
            **cognify_kwargs,
        )
    return captured["scope"], captured["resolved"]


@pytest.mark.asyncio
async def test_cognify_resolver_notes_chunking_on_the_pipeline_run_scope(fake_capture_sink):
    scope, (tasks,) = await _run_cognify_resolving(
        [_text_item()], chunker=TextChunker, chunk_size=321
    )

    # The standard list was resolved and its chunking task is what got noted.
    names = [task.executable.__name__ for task in tasks]
    assert names[:2] == ["classify_documents", "extract_chunks_from_documents"]
    assert scope.fields["chunking.chunker"] == "TextChunker"
    assert scope.fields["chunking.chunk_size"] == 321

    # ... and the note reaches the pipeline manifest the scope emits on exit.
    await capture.drain()
    [manifest] = [
        record
        for record in fake_capture_sink.records
        if record["kind"] == capture.KIND_RUN_MANIFEST
    ]
    assert manifest["payload"]["kind"] == "pipeline"
    assert manifest["payload"]["chunking.chunker"] == "TextChunker"
    assert manifest["payload"]["chunking.chunk_size"] == 321


@pytest.mark.asyncio
async def test_cognify_resolver_notes_nothing_for_a_code_item(fake_capture_sink):
    """The note follows the RESOLVED list: a code item runs the enola list, which
    chunks nothing, so no chunking fields appear even though a standard list (with
    a chunking task) was built up front."""
    scope, (tasks,) = await _run_cognify_resolving([_code_item()], chunk_size=321)

    assert [task.executable.__name__ for task in tasks] == ["extract_code_files_graph"]
    assert "chunking.chunker" not in scope.fields
    assert "chunking.chunk_size" not in scope.fields


@pytest.mark.asyncio
async def test_cognify_resolver_calls_the_note_helper_once_per_item(fake_capture_sink):
    """Placement contract: _note_chunking_config is invoked by the resolver, once per
    resolved item, with the list that item resolved to — not at construction time."""
    spy = MagicMock(wraps=cognify_module._note_chunking_config)

    with patch.object(cognify_module, "_note_chunking_config", spy):
        _scope, resolved = await _run_cognify_resolving(
            [_text_item(), _code_item()], chunk_size=321
        )

    assert spy.call_count == 2
    assert [call.args[0] for call in spy.call_args_list] == resolved
    assert capture.current_scope() is None  # the fake executor's scope has closed
