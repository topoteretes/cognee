"""Integration test for figma_source through cognee.add(): verifies Data records, metadata, and node sets."""

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import figma_source
from cognee.tests.unit.tasks.connectors.fake_figma import FakeFigma

DATASET = "figma_integration_test"
TABLE = "figma_test"


async def _add(source) -> None:
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _documents() -> dict[str, Data]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        str(data.system_metadata["external_id"]): data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict) and data.system_metadata.get("source") == "figma"
    }


def _node_set(document: Data) -> list[str] | None:
    return (document.external_metadata or {}).get("node_set")


@pytest.mark.asyncio
async def test_figma_add_ingests_data_records_with_structure_and_nodeset():
    fake = FakeFigma()
    fake.add_file(
        "file1",
        name="Design System",
        pages=[
            {
                "id": "1:1",
                "name": "Buttons Page",
                "type": "CANVAS",
                "children": [
                    {
                        "id": "1:2",
                        "name": "Button",
                        "type": "TEXT",
                        "characters": "Submit",
                    }
                ],
            }
        ],
    )

    source = figma_source(
        resource_name=TABLE,
        file_keys=["file1"],
        service=fake,
    )
    await _add(source)

    docs = await _documents()
    assert "file:file1" in docs
    assert "page:file1:1:1" in docs

    file_doc = docs["file:file1"]
    page_doc = docs["page:file1:1:1"]

    assert file_doc.system_metadata["source"] == "figma"
    assert page_doc.system_metadata["source"] == "figma"

    # Node set check
    assert _node_set(file_doc) == ["figma:file1"]
    assert _node_set(page_doc) == ["figma:file1"]

    # Structure check on page document
    assert page_doc.system_metadata["structure"] == {
        "ancestors": [
            {
                "kind": "file",
                "id": "file:file1",
                "name": "Design System",
                "document": True,
            }
        ]
    }
