"""Integration test for figma_source forget-on-delete behavior."""

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import figma_source
from cognee.tests.unit.tasks.connectors.fake_figma import FakeFigma

DATASET = "figma_forget_test"
TABLE = "figma_forget_table"


async def _sync(fake: FakeFigma):
    source = figma_source(
        resource_name=TABLE,
        file_keys=["file1"],
        service=fake,
    )
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    return source


async def _documents() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        data.system_metadata["external_id"]: data.id
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict) and data.system_metadata.get("source") == "figma"
    }


@pytest.mark.asyncio
async def test_figma_forget_deletes_removed_canvas_document():
    fake = FakeFigma()
    fake.add_file(
        "file1",
        name="Design",
        pages=[
            {"id": "p1", "name": "Page 1", "type": "CANVAS", "children": []},
            {"id": "p2", "name": "Page 2", "type": "CANVAS", "children": []},
        ],
        last_modified="2026-10-31T08:00:00Z",
    )

    # First sync: both pages present
    await _sync(fake)
    docs_first = await _documents()
    assert "file:file1" in docs_first
    assert "page:file1:p1" in docs_first
    assert "page:file1:p2" in docs_first

    # Page 2 is removed in Figma
    fake.files["file1"]["lastModified"] = "2026-10-31T09:00:00Z"
    fake.files["file1"]["document"]["children"] = [
        {"id": "p1", "name": "Page 1", "type": "CANVAS", "children": []}
    ]

    # Second sync: Page 2 is forgotten
    await _sync(fake)
    docs_second = await _documents()
    assert "file:file1" in docs_second
    assert "page:file1:p1" in docs_second
    assert "page:file1:p2" not in docs_second
