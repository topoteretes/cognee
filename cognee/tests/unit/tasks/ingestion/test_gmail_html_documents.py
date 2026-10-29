"""Gmail inline HTML facts survive real dlt staging and document mapping."""

import base64
from uuid import NAMESPACE_OID, uuid5

import pytest

from cognee.tasks.ingestion.connectors.gmail import parse_message
from cognee.tasks.ingestion.resolve_dlt_sources import _build_document_data_item


@pytest.mark.asyncio
async def test_html_body_facts_survive_staging_and_document_mapping(staging):
    message = {
        "id": "orion-review",
        "snippet": "The review plan has changed.",
        "payload": {
            "mimeType": "text/html",
            "headers": [{"name": "Subject", "value": "Orion review"}],
            "body": {
                "data": base64.urlsafe_b64encode(
                    b"<p>The review plan has changed.</p>"
                    b"<p>Review: November 12, 2026 at 14:30 UTC. Owner: Mira Chen.</p>"
                ).decode()
            },
        },
    }
    rows = await staging("gmail_html_messages", [parse_message(message)], tag="gmail")
    assert len(rows) == 1
    assert "November 12, 2026 at 14:30 UTC" in rows[0].row_data["content"]

    item = _build_document_data_item(rows[0], uuid5(NAMESPACE_OID, message["id"]), "gmail")
    assert "Owner: Mira Chen." in item.data
    assert item.data.startswith("# Orion review")
    assert item.literal_text is True
    assert item.system_metadata["external_id"] == message["id"]
