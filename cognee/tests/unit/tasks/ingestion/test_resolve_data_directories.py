from pathlib import Path

import pytest

from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories


@pytest.mark.asyncio
async def test_resolve_data_directories_expands_file_uri_directory(tmp_path):
    root = tmp_path / "docs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    first_file = root / "a.txt"
    second_file = nested / "b.txt"
    first_file.write_text("a", encoding="utf-8")
    second_file.write_text("b", encoding="utf-8")

    result = await resolve_data_directories(root.as_uri())

    assert sorted(Path(item).name for item in result) == ["a.txt", "b.txt"]
    assert all(not item.startswith("file://") for item in result)


@pytest.mark.asyncio
async def test_resolve_data_directories_expands_file_uri_directory_non_recursive(tmp_path):
    root = tmp_path / "docs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    first_file = root / "a.txt"
    second_file = nested / "b.txt"
    first_file.write_text("a", encoding="utf-8")
    second_file.write_text("b", encoding="utf-8")

    result = await resolve_data_directories(root.as_uri(), include_subdirectories=False)

    assert [Path(item).name for item in result] == ["a.txt"]
