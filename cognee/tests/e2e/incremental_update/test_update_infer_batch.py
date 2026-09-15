"""update() without a data_id, and update() over a batch, on the default stack.

Runs in a scratch root with the mocked LLM and embeddings (same fixture shape
as test_add_existing_document). The rules under test: a local file is matched
by its path and an upload by its filename; raw text and a
renamed file are refused with the way to find the id, and nothing is written;
a directory or a list updates one document per file and reports the counts,
carrying on past a document whose update fails.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)


@pytest.fixture(scope="module")
def infer_env():
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_update_infer_"))
    import cognee

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL=os.environ.get("INCR_TEST_ACL", "true"),
    )
    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        (
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            "create_embedding_engine",
        ),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    import re

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    marker = re.compile(r"ENT[A-Z0-9]+")

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(marker.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=f"marker {n}") for n in names],
                edges=[],
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        return response_model() if isinstance(response_model, type) else "mock"

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate
    yield root
    LLMGateway.acreate_structured_output = original
    shutil.rmtree(root, ignore_errors=True)


def _upload(content: bytes, filename: str):
    from starlette.datastructures import UploadFile

    spooled = tempfile.SpooledTemporaryFile()  # noqa: SIM115 - handed to UploadFile, which owns it
    spooled.write(content)
    spooled.seek(0)
    return UploadFile(file=spooled, filename=filename)


async def _rows(dataset_id):
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data

    return {row.name: row for row in await get_dataset_data(dataset_id)}


@pytest.mark.asyncio
async def test_update_infers_the_document_from_a_path_or_an_upload(infer_env):
    await reset_backend_state()
    import cognee
    from cognee.api.v1.exceptions import UpdateTargetNotInferredError
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.users.methods import get_default_user

    report = infer_env / "report.txt"
    report.write_text("Quarterly report ENTREPORT, version one.\n\nSecond paragraph ENTTWO.\n")
    await cognee.add(str(report), dataset_name="infer")
    await cognee.add(
        _upload(b"Meeting notes ENTNOTES, version one.\n\nMore notes ENTMORE.\n", "notes.txt"),
        dataset_name="infer",
    )
    await cognee.cognify(datasets=["infer"])
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "infer")
    rows = await _rows(dataset.id)
    assert set(rows) == {"report", "notes"}
    report_id, notes_id = rows["report"].id, rows["notes"].id

    # The same path, edited: the document is found by its location.
    report.write_text(
        "Quarterly report ENTREPORT ENTV2, version two.\n\nSecond paragraph ENTTWO.\n"
    )
    result = await cognee.update(str(report), dataset.id, user=user)
    assert (result["status"], result["data_id"]) == ("incremental", report_id), result
    assert result["regions"] == 1 and result["total_chunks"] >= 1

    # The same path, unchanged: still found, nothing to do.
    result = await cognee.update(str(report), dataset.id, user=user)
    assert (result["status"], result["data_id"]) == ("unchanged", report_id), result

    # An upload under the stored filename: found by name.
    result = await cognee.update(
        _upload(
            b"Meeting notes ENTNOTES ENTV2, version two.\n\nMore notes ENTMORE.\n", "notes.txt"
        ),
        dataset.id,
        user=user,
    )
    assert (result["status"], result["data_id"]) == ("incremental", notes_id), result

    # Raw text has no origin: refused, with the way to find the id.
    with pytest.raises(UpdateTargetNotInferredError) as refused:
        await cognee.update("Quarterly report ENTREPORT, version four.\n", dataset.id, user=user)
    assert refused.value.status_code == 422
    assert refused.value.unresolved[0]["reason"] == "raw text has no origin to match"
    assert "cognee.datasets.list_data(dataset_id)" in refused.value.message
    assert f"GET /api/v1/datasets/{dataset.id}/data" in refused.value.api_message

    # A renamed copy is a document no row came from: refused, nothing written.
    renamed = infer_env / "report_final.txt"
    renamed.write_text("Quarterly report ENTREPORT ENTV4, version four.\n")
    with pytest.raises(UpdateTargetNotInferredError) as refused:
        await cognee.update(str(renamed), dataset.id, user=user)
    assert refused.value.unresolved == [
        {"input": "report_final.txt", "reason": "no document in the dataset came from it"}
    ]
    after = await _rows(dataset.id)
    assert set(after) == {"report", "notes"} and after["report"].id == report_id

    # With the id, the renamed file is the way to move a document: it works.
    result = await cognee.update(str(renamed), dataset.id, data_id=report_id, user=user)
    assert (result["status"], result["data_id"]) == ("incremental", report_id), result
    assert len(await _rows(dataset.id)) == 2, "update() never mints a row"


@pytest.mark.asyncio
async def test_a_directory_updates_every_file_and_reports_the_counts(infer_env, monkeypatch):
    await reset_backend_state()
    import cognee
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.users.methods import get_default_user

    folder = infer_env / "folder"
    folder.mkdir(exist_ok=True)
    (folder / "a.txt").write_text("Folder file A ENTA.\n\nMore A ENTAA.\n")
    (folder / "b.txt").write_text("Folder file B ENTB.\n\nMore B ENTBB.\n")
    (folder / "c.txt").write_text("Folder file C ENTC.\n\nMore C ENTCC.\n")
    await cognee.add(str(folder), dataset_name="batch")
    await cognee.cognify(datasets=["batch"])
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "batch")
    rows = await _rows(dataset.id)
    assert set(rows) == {"a", "b", "c"}

    (folder / "a.txt").write_text("Folder file A ENTA ENTEDIT.\n\nMore A ENTAA.\n")
    (folder / "c.txt").write_text("Folder file C ENTC ENTEDIT.\n\nMore C ENTCC.\n")
    result = await cognee.update(str(folder), dataset.id, user=user)

    assert (result["status"], result["total"]) == ("completed", 3), result
    assert (result["updated"], result["unchanged"], result["failed"]) == (2, 1, 0)
    assert [r["data_id"] for r in result["results"]] == [rows[n].id for n in ("a", "b", "c")]
    assert [r["status"] for r in result["results"]] == ["incremental", "unchanged", "incremental"]
    assert len(await _rows(dataset.id)) == 3

    # One document's failure is recorded and the batch carries on.
    import cognee.api.v1.update.update  # bind the real submodule

    update_module = sys.modules["cognee.api.v1.update.update"]
    real_engine = update_module.incremental_update

    async def failing_for_b(**kwargs):
        if kwargs["data_id"] == rows["b"].id:
            raise RuntimeError("simulated extraction failure")
        return await real_engine(**kwargs)

    monkeypatch.setattr(update_module, "incremental_update", failing_for_b)
    (folder / "a.txt").write_text("Folder file A ENTA ENTEDIT ENTAGAIN.\n\nMore A ENTAA.\n")
    (folder / "b.txt").write_text("Folder file B ENTB ENTEDIT.\n\nMore B ENTBB.\n")
    result = await cognee.update(
        [str(folder / "a.txt"), str(folder / "b.txt"), str(folder / "c.txt")],
        dataset.id,
        user=user,
    )
    monkeypatch.undo()

    assert (result["status"], result["updated"], result["unchanged"]) == ("partial", 1, 1)
    assert result["failed"] == 1
    failed = result["results"][1]
    assert (failed["status"], failed["data_id"]) == ("failed", rows["b"].id)
    assert failed["error"] == {
        "error_class": "RuntimeError",
        "message": "simulated extraction failure",
    }

    # The failed document is retried on its own, by the id the result carries.
    result = await cognee.update(
        str(folder / "b.txt"), dataset.id, data_id=failed["data_id"], user=user
    )
    assert (result["status"], result["data_id"]) == ("incremental", rows["b"].id), result
