"""What a Gmail connection can enumerate, given an access token.

Split from cognee.modules.integrations.gmail.adapter (cognee dev, 2026-10-26):
:func:`list_resources` is the body of ``GoogleGmailIntegration.list_resources`` there,
taken out of the host adapter so the contract object (``connector.py``) can offer the same
picker without a credential row. The entries keep the shape cognee's generic picker reads
(``id``, ``name``, ``description``, ``attributes``); the contract object wraps them in
``ConnectorResource``. Pure: nothing here imports cognee.
"""

from typing import Any

from cognee_connector_gmail import client as gmail_client


async def list_resources(access_token: str) -> list[dict[str, Any]]:
    """The mailbox's labels, system and user-made, for a picker."""
    labels = await gmail_client.list_labels(access_token)
    return [
        {
            "id": str(label["id"]),
            "name": str(label.get("name") or label["id"]),
            "description": None,
            "attributes": {
                "type": label.get("type"),
                "messages_total": label.get("messagesTotal"),
            },
        }
        for label in labels.get("labels", []) or []
        if label.get("id")
    ]
