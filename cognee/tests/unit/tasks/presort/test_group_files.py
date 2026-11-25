import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.tasks.presort.classify_files import classify_files
from cognee.tasks.presort.group_files import group_files, sanitize_dataset_name
from cognee.tasks.presort.models import FileRecord
from cognee.tasks.presort.scan_folder import scan_folder


def test_sanitize_dataset_name():
    assert sanitize_dataset_name("My Tool (v2)") == "my_tool_v2"
    assert sanitize_dataset_name("  ") == "unsorted"
    assert sanitize_dataset_name("docs.finance") == "docs.finance"


@pytest.mark.asyncio
async def test_grouping_layers(messy_folder):
    files, _ = await scan_folder(messy_folder)
    await classify_files(files)
    groups = await group_files(messy_folder, files, dataset_prefix="dl_")

    by_name = {group.name: group for group in groups}

    # Code project wins over folder grouping.
    assert by_name["my_tool"].kind == "code_project"
    assert any("main.py" in path for path in by_name["my_tool"].file_paths)

    # Subdirectory becomes a folder group.
    assert by_name["invoices"].kind == "folder"
    assert len(by_name["invoices"].file_paths) == 2

    # Loose root files fall back to extension families.
    assert by_name["documents"].kind == "extension_family"
    assert by_name["images"].kind == "extension_family"

    # Prefix flows into dataset names.
    assert by_name["invoices"].dataset_name == "dl_invoices"

    # Every kept file lands in exactly one group.
    grouped_paths = [path for group in groups for path in group.file_paths]
    assert sorted(grouped_paths) == sorted(record.path for record in files)


@pytest.mark.asyncio
async def test_classify_families(messy_folder):
    files, _ = await scan_folder(messy_folder)
    await classify_files(files)
    by_name = {record.name: record for record in files}

    assert by_name["report.pdf"].family == "documents"
    assert by_name["holiday.jpg"].family == "images"
    assert by_name["main.py"].family == "code"
    assert by_name["notes.txt"].family == "documents"


@pytest.mark.asyncio
async def test_llm_classify_decodes_utf16_sample(tmp_path):
    """Regression for #6563: once is_text_content admits UTF-16 files, the LLM

    classifier reached them too (it filters on is_text). A raw UTF-8 decode of
    the sample fed the classifier interleaved null bytes; decoding by the BOM
    must give it the real characters.
    """
    path = tmp_path / "contract.txt"
    path.write_text("This Services Agreement is entered into by ACME Corp.", encoding="utf-16")
    record = FileRecord(
        path=str(path), name="contract.txt", extension="txt", is_text=True, size_bytes=1
    )

    # Patch the submodule object: the package re-exports a function under the
    # submodule's name, so a dotted target resolves to the function on 3.10.
    extraction_module = importlib.import_module(
        "cognee.infrastructure.llm.extraction.extract_categories"
    )
    with patch.object(
        extraction_module, "extract_categories", new=AsyncMock(return_value=None)
    ) as llm_mock:
        await classify_files([record], use_llm=True)

    sample = llm_mock.await_args.args[0]
    assert "ACME Corp" in sample  # decoded text, not null-laden bytes
    assert "\x00" not in sample
    assert record.warnings == []
