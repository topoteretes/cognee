"""add() refuses a file it already holds with other content and points at update().

Runs on the default local stack in a scratch root with the mocked LLM and
embeddings (same fixture shape as test_incremental_update). The rule under
test is the resolution step of ingestion, so the scenarios are about which
call is accepted, which is refused, and that a refusal writes nothing.
"""

import io
import shutil
import tempfile
from pathlib import Path

import pytest

from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)


@pytest.fixture(scope="module")
def add_env():
    import os

    root = Path(tempfile.mkdtemp(prefix="cognee_add_existing_"))
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
        # Documents produce entities; the same deterministic extraction the
        # other suites in this directory use, so the graph is never empty.
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


@pytest.mark.asyncio
async def test_re_adding_a_changed_file_is_refused_and_update_is_the_way(add_env):
    await reset_backend_state()
    import cognee
    from cognee.api.v1.exceptions import DocumentUpdateRequiredError
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    report = add_env / "report.txt"
    report.write_text("Quarterly report ENTREPORT, version one.\n")
    await cognee.add(str(report), dataset_name="existing")
    await cognee.add(
        _upload(b"Meeting notes ENTNOTES, version one.\n", "notes.txt"), dataset_name="existing"
    )
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "existing")
    rows = {row.name: row for row in await get_dataset_data(dataset.id)}
    assert set(rows) == {"report", "notes"}

    # Identical content re-added: a no-op, still two rows.
    await cognee.add(str(report), dataset_name="existing")
    await cognee.add(
        _upload(b"Meeting notes ENTNOTES, version one.\n", "notes.txt"), dataset_name="existing"
    )
    assert len(await get_dataset_data(dataset.id)) == 2

    # The same path with new content: refused, naming the document to update.
    report.write_text("Quarterly report ENTREPORT ENTV2, version two.\n")
    with pytest.raises(DocumentUpdateRequiredError) as refused:
        await cognee.add(str(report), dataset_name="existing")
    assert refused.value.status_code == 409
    assert refused.value.conflicts == [{"name": "report.txt", "data_id": rows["report"].id}]
    assert "cognee.update(" in refused.value.message
    assert "PATCH /api/v1/update" in refused.value.api_message

    # The same upload name with new content: refused too.
    with pytest.raises(DocumentUpdateRequiredError) as refused:
        await cognee.add(
            _upload(b"Meeting notes ENTNOTES ENTV2, version two.\n", "notes.txt"),
            dataset_name="existing",
        )
    assert refused.value.conflicts[0]["data_id"] == rows["notes"].id

    # A batch with one changed file is refused whole, before anything is written.
    fresh = add_env / "fresh.txt"
    fresh.write_text("A brand new document ENTFRESH.\n")
    with pytest.raises(DocumentUpdateRequiredError):
        await cognee.add([str(fresh), str(report)], dataset_name="existing")
    assert len(await get_dataset_data(dataset.id)) == 2, "a refused add must write nothing"

    # The way to do it: update() keeps the id and takes the new content.
    result = await cognee.update(str(report), dataset.id, data_id=rows["report"].id, user=user)
    assert result["data_id"] == rows["report"].id
    assert result["status"] in ("incremental", "full_rebuild"), result
    after = {row.name: row for row in await get_dataset_data(dataset.id)}
    assert after["report"].id == rows["report"].id and len(after) == 2

    # A directory re-added after one of its files changed is refused whole too.
    folder = add_env / "folder"
    folder.mkdir(exist_ok=True)
    (folder / "a.txt").write_text("Folder file A ENTA.\n")
    (folder / "b.txt").write_text("Folder file B ENTB.\n")
    await cognee.add(str(folder), dataset_name="existing")
    assert len(await get_dataset_data(dataset.id)) == 4
    (folder / "b.txt").write_text("Folder file B ENTB ENTEDIT, edited.\n")
    with pytest.raises(DocumentUpdateRequiredError) as refused:
        await cognee.add(str(folder), dataset_name="existing")
    assert [c["name"] for c in refused.value.conflicts] == ["b.txt"]
    assert len(await get_dataset_data(dataset.id)) == 4

    # Same name in a different dataset is a different document, not a conflict.
    await cognee.add(str(report), dataset_name="another")
    # And another user's dataset is out of scope by construction (dedup scope),
    # which the query shares with identify_many.
