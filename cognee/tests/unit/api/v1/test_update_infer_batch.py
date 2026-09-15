"""update() infers the document from the input's origin and takes batches.

Without a data_id, a local file is matched by its path and an upload by its
filename against the dataset's documents; whatever cannot be matched is
refused, with the way to find the id, before anything is written. A list, or
a directory, updates one document per input and answers with the per-document
results and their counts (SDK-587).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import cognee.api.v1.update.update  # bind the real submodule
import cognee.modules.ingestion.identify_by_origin  # bind the real submodule
from cognee.api.v1.exceptions import UpdateTargetNotInferredError
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.ingestion.identify_by_origin import Origin, OriginMatch
from cognee.tasks.ingestion.data_item import DataItem

update_module = sys.modules["cognee.api.v1.update.update"]
origin_module = sys.modules["cognee.modules.ingestion.identify_by_origin"]
data_methods_module = sys.modules["cognee.modules.data.methods"]

pytestmark = pytest.mark.asyncio


def _user():
    return SimpleNamespace(id=uuid4(), tenant_id=None)


def _summary(status="incremental"):
    changed = status == "incremental"
    return {
        "status": status,
        "regions": int(changed),
        "deleted_chunks": int(changed),
        "added_chunks": int(changed),
        "reused_chunks": 0,
        "kept_chunks": 3,
        "reindexed_chunks": 0,
        "total_chunks": 3 + int(changed),
        "pipeline_run_id": uuid4() if changed else None,
    }


def _origin(label, **fields):
    return Origin(label=label, content_hash=f"hash-{label}", **fields)


def _key(item):
    # Inputs are keyed by basename so a directory's expanded paths match too.
    return Path(item).name if isinstance(item, str) else item


class _Stack:
    """update() with the engine, id resolution and the origin lookup stubbed."""

    def __init__(self, incremental, origins=None, matches=None):
        self.incremental = incremental
        self.resolve = AsyncMock(side_effect=lambda dataset_id, data_id: data_id)
        origins = origins or {}
        matches = matches or {}
        self.origin_of = AsyncMock(side_effect=lambda item: origins.get(_key(item)))
        self.find_by_origin = AsyncMock(
            side_effect=lambda found, user, dataset_id: {o: matches.get(o, []) for o in found}
        )
        self.patches = (
            patch.object(update_module, "incremental_update", incremental),
            patch.object(data_methods_module, "resolve_data_id", self.resolve),
            patch.object(origin_module, "origin_of", self.origin_of),
            patch.object(origin_module, "find_by_origin", self.find_by_origin),
        )

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        return False


async def test_the_document_is_inferred_from_the_inputs_origin():
    dataset_id, doc = uuid4(), uuid4()
    origin = _origin("report.txt", location="file:///docs/report.txt")
    stack = _Stack(
        AsyncMock(return_value=_summary()),
        origins={"report.txt": origin},
        matches={origin: [OriginMatch(data_id=doc, content_hash="old")]},
    )

    with stack:
        result = await update_module.update("/docs/report.txt", dataset_id, user=_user())

    assert (result["status"], result["data_id"]) == ("incremental", doc)
    assert stack.incremental.await_args.kwargs["data_id"] == doc
    stack.resolve.assert_not_awaited()


async def test_an_explicit_data_id_is_resolved_and_nothing_is_inferred():
    dataset_id, doc = uuid4(), uuid4()
    stack = _Stack(AsyncMock(return_value=_summary()))

    with stack:
        result = await update_module.update("edited text", dataset_id, data_id=doc, user=_user())

    assert result["data_id"] == doc
    stack.resolve.assert_awaited_once_with(dataset_id, doc)
    stack.origin_of.assert_not_awaited()


async def test_raw_text_without_data_id_is_refused_with_the_way_to_find_it():
    dataset_id = uuid4()
    stack = _Stack(AsyncMock())

    with stack, pytest.raises(UpdateTargetNotInferredError) as refused:
        await update_module.update("edited text", dataset_id, user=_user())

    assert refused.value.status_code == 422
    assert refused.value.unresolved == [
        {"input": "text 'edited text'", "reason": "raw text has no origin to match"}
    ]
    assert "cognee.datasets.list_data(dataset_id)" in refused.value.message
    assert f"dataset_id={dataset_id}, data_id=<data_id>" in refused.value.message
    assert f"GET /api/v1/datasets/{dataset_id}/data" in refused.value.api_message
    stack.incremental.assert_not_awaited()


async def test_unmatched_and_ambiguous_origins_are_refused_together_before_any_work():
    dataset_id = uuid4()
    moved = _origin("moved.txt", location="file:///new/moved.txt")
    shared = _origin("notes.txt", name=("notes", "txt"))
    fine = _origin("fine.txt", location="file:///docs/fine.txt")
    stack = _Stack(
        AsyncMock(return_value=_summary()),
        origins={"fine.txt": fine, "moved.txt": moved, "notes.txt": shared},
        matches={
            fine: [OriginMatch(data_id=uuid4(), content_hash="a")],
            shared: [
                OriginMatch(data_id=uuid4(), content_hash="b"),
                OriginMatch(data_id=uuid4(), content_hash="c"),
            ],
        },
    )

    with stack, pytest.raises(UpdateTargetNotInferredError) as refused:
        await update_module.update(
            ["/docs/fine.txt", "/new/moved.txt", "/up/notes.txt"], dataset_id, user=_user()
        )

    assert refused.value.unresolved == [
        {"input": "moved.txt", "reason": "no document in the dataset came from it"},
        {"input": "notes.txt", "reason": "2 documents came from it"},
    ]
    stack.incremental.assert_not_awaited(), "the matched input is not updated either"


async def test_a_list_answers_with_per_document_results_and_counts():
    dataset_id = uuid4()
    ids = [uuid4() for _ in range(3)]
    engine = AsyncMock(
        side_effect=[
            _summary("incremental"),
            _summary("unchanged"),
            RuntimeError("extraction down"),
        ]
    )
    stack = _Stack(engine)

    with stack:
        result = await update_module.update(
            [DataItem(data=f"doc {i}", data_id=ids[i]) for i in range(3)], dataset_id, user=_user()
        )

    counts = (result["status"], result["total"], result["updated"], result["unchanged"])
    assert counts == ("partial", 3, 1, 1) and result["failed"] == 1
    assert [r["data_id"] for r in result["results"]] == ids
    assert [r["status"] for r in result["results"]] == ["incremental", "unchanged", "failed"]
    assert result["results"][2]["error"] == {
        "error_class": "RuntimeError",
        "message": "extraction down",
    }
    assert result["dataset_id"] == dataset_id and result["duration_seconds"] >= 0
    assert stack.incremental.await_count == 3, "one failure does not stop the batch"


async def test_a_one_item_list_is_that_document_and_its_failure_raises():
    """The shape the HTTP router always sent keeps its single-document contract."""
    dataset_id, doc = uuid4(), uuid4()
    stack = _Stack(AsyncMock(return_value=_summary()))
    with stack:
        result = await update_module.update(["doc"], dataset_id, data_id=doc, user=_user())
    assert result["status"] == "incremental" and "results" not in result

    stack = _Stack(AsyncMock(side_effect=RuntimeError("down")))
    with stack, pytest.raises(RuntimeError, match="down"):
        await update_module.update([DataItem(data="doc", data_id=doc)], dataset_id, user=_user())


async def test_every_document_failing_is_a_failed_batch():
    dataset_id, ids = uuid4(), [uuid4(), uuid4()]
    stack = _Stack(AsyncMock(side_effect=RuntimeError("down")))

    with stack:
        result = await update_module.update(
            [DataItem(data=f"doc {i}", data_id=ids[i]) for i in range(2)], dataset_id, user=_user()
        )

    assert (result["status"], result["total"], result["failed"]) == ("failed", 2, 2)
    assert [r["data_id"] for r in result["results"]] == ids


async def test_a_directory_is_a_batch_of_its_files(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("A")
    (folder / "b.txt").write_text("B")
    a = _origin("a.txt", location=(folder / "a.txt").as_uri())
    b = _origin("b.txt", location=(folder / "b.txt").as_uri())
    ids = {a: uuid4(), b: uuid4()}
    stack = _Stack(
        AsyncMock(return_value=_summary()),
        origins={"a.txt": a, "b.txt": b},
        matches={o: [OriginMatch(data_id=i, content_hash="x")] for o, i in ids.items()},
    )

    with stack:
        result = await update_module.update(str(folder), uuid4(), user=_user())

    assert (result["total"], result["updated"]) == (2, 2)
    assert [r["data_id"] for r in result["results"]] == [ids[a], ids[b]]


async def test_two_inputs_targeting_one_document_are_refused():
    dataset_id, doc = uuid4(), uuid4()
    stack = _Stack(AsyncMock(return_value=_summary()))

    with stack, pytest.raises(IngestionError, match="both target document"):
        await update_module.update(
            [DataItem(data="v1", data_id=doc), DataItem(data="v2", data_id=doc)],
            dataset_id,
            user=_user(),
        )

    stack.incremental.assert_not_awaited()


async def test_a_uuid_passed_as_data_names_the_argument_order():
    with pytest.raises(TypeError, match=r"update\(data, dataset_id, data_id=\.\.\.\)"):
        await update_module.update(uuid4(), uuid4(), uuid4())
