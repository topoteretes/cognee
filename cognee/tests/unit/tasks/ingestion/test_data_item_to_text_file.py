from pathlib import Path

import pytest

from cognee.tasks.ingestion.data_item_to_text_file import data_item_to_text_file


@pytest.mark.asyncio
async def test_file_uri_loads_local_text_file(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello text")

    content, loader = await data_item_to_text_file(
        Path(file_path).as_uri(),
        preferred_loaders={"text_loader": {"persist": False}},
    )

    assert content == "hello text"
    assert loader.loader_name == "text_loader"
