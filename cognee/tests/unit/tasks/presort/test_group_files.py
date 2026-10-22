import pytest

from cognee.tasks.presort.classify_files import classify_files
from cognee.tasks.presort.group_files import group_files, sanitize_dataset_name
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
