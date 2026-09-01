"""cognify's chunking manifest notes (SDK-529): the per-item task resolver records the
chunker and chunk size on the active run scope, read off the task list it resolved."""

import importlib
from uuid import uuid4

import pytest

from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.observability import capture
from cognee.modules.pipelines.tasks.task import Task
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents

cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")

pytestmark = pytest.mark.usefixtures("capture_reset")


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
