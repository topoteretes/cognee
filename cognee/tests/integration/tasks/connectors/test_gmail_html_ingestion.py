"""Gmail HTML and explicit refresh through real add()/document storage, no LLM."""

from pathlib import Path

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import gmail as gmail_module
from cognee.tasks.ingestion.connectors import gmail_source
from cognee.tests.unit.tasks.connectors.test_gmail_source import (
    FakeGmailService,
    _b64,
    _make_message,
)


@pytest.mark.asyncio
async def test_html_refresh_replaces_stored_preview_and_deletion_forgets_it(
    clean_environment, monkeypatch
):
    dataset_name = "gmail_html_integration"
    message = _make_message("orion", subject="Orion review")
    message["snippet"] = "The review plan changed."
    message["payload"].update(
        mimeType="text/html",
        body={"data": _b64("<p>Review: November 12 at 14:30 UTC.</p><p>Owner: Mira Chen.</p>")},
    )
    service = FakeGmailService(messages=[message])

    async def sync(**options):
        await cognee.add(
            gmail_source(service=service, resource_name="gmail_html_account", **options),
            dataset_name=dataset_name,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )

    async def documents():
        datasets = await get_authorized_existing_datasets(
            user=await get_default_user(), permission_type="write", datasets=[dataset_name]
        )
        return await get_dataset_data(datasets[0].id)

    def text(document):
        return Path(document.raw_data_location.removeprefix("file://")).read_text(encoding="utf-8")

    parse = gmail_module.parse_message
    with monkeypatch.context() as legacy:
        legacy.setattr(
            gmail_module, "parse_message", lambda item: {**parse(item), "content": item["snippet"]}
        )
        await sync()
    before = (await documents())[0]
    assert "Mira Chen" not in text(before)

    await sync()
    assert (await documents())[0].id == before.id
    await sync(full_refresh=True)
    stored = await documents()
    assert len(stored) == 1
    assert stored[0].system_metadata["source"] == "gmail"
    assert stored[0].system_metadata["external_id"] == "orion"
    assert "November 12 at 14:30 UTC" in text(stored[0])
    assert "Owner: Mira Chen." in text(stored[0])

    await sync()
    assert (await documents())[0].id == stored[0].id
    service.history_response = {
        "history": [{"id": "600", "messagesDeleted": [{"message": {"id": "orion"}}]}],
        "historyId": "600",
    }
    await sync()
    assert await documents() == []
