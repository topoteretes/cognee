"""LlamaIndex text saves keep storage provenance; caller paths do not."""

import importlib
from pathlib import Path

import pytest

from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path
from cognee.infrastructure.loaders.LoaderInterface import LoaderResult
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.ingestion.data_item_to_text_file import data_item_to_text_file
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage_detailed

llama_core = pytest.importorskip("llama_index.core")
llama_schema = pytest.importorskip("llama_index.core.schema")
Document = llama_core.Document
ImageDocument = llama_schema.ImageDocument
save_module = importlib.import_module("cognee.modules.ingestion.save_data_to_file")
loader_module = importlib.import_module("cognee.tasks.ingestion.data_item_to_text_file")


@pytest.fixture
def local_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        save_module, "get_storage_config", lambda: {"data_root_directory": str(tmp_path)}
    )
    monkeypatch.setattr(loader_module.settings, "accept_local_file_path", False)
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("document_type", [Document, ImageDocument])
async def test_llama_index_text_is_loadable_with_local_paths_disabled(local_storage, document_type):
    stored = await save_data_item_to_storage_detailed(document_type(text="LlamaIndex text."))
    assert stored.is_internal_storage is True

    loader_result, _ = await data_item_to_text_file(
        get_data_file_path(stored.file_path), is_internal_storage=stored.is_internal_storage
    )
    assert isinstance(loader_result, LoaderResult)
    assert (
        Path(get_data_file_path(loader_result.file_path)).read_text(encoding="utf-8")
        == "LlamaIndex text."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("document_type", [Document, ImageDocument])
async def test_llama_index_caller_paths_remain_untrusted(local_storage, document_type):
    source = local_storage / "caller.txt"
    source.write_text("Caller-provided path.", encoding="utf-8")
    if document_type is Document:
        document = Document(
            text="Unused text.",
            metadata={"file_path": str(source), "is_internal_storage": True},
        )
    else:
        from PIL import Image

        source = local_storage / "caller.png"
        Image.new("RGB", (1, 1)).save(source)
        document = ImageDocument(text="Unused text.", image_path=str(source))

    stored = await save_data_item_to_storage_detailed(document)
    assert stored.is_internal_storage is False
    with pytest.raises(IngestionError, match="Local files are not accepted"):
        await data_item_to_text_file(
            stored.file_path, is_internal_storage=stored.is_internal_storage
        )
