import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cognee_community_connector_zotero.zotero as zotero_module
from cognee_community_connector_zotero.zotero import ZoteroClient, iter_zotero_rows


class FakeClient(ZoteroClient):
    page_size = 3

    def __init__(self):
        super().__init__(api_key="key", user_id="42", base_url="https://zotero.test")
        self.calls = []

    def get_json(self, path, params=None):
        self.calls.append((path, params))
        if path == "items":
            if params.get("start") == self.page_size:
                return ([], 11)
            return (
                [
                    {
                        "data": {
                            "key": "A1",
                            "itemType": "journalArticle",
                            "title": "Versioned Sync",
                            "abstractNote": "Exact library versions.",
                            "creators": [{"firstName": "Ada", "lastName": "Lovelace"}],
                            "version": 8,
                        }
                    },
                    {"data": {"key": "N1", "itemType": "note", "note": "<p>Readable note</p>"}},
                    {"data": {"key": "F1", "itemType": "attachment", "filename": "paper.pdf"}},
                ],
                11,
            )
        if path == "items/F1/fulltext":
            return ({"content": "Attachment body"}, None)
        if path == "deleted":
            return ({"items": ["OLD"]}, 12)
        raise AssertionError(path)


def test_iter_zotero_rows_flattens_selected_content():
    client = FakeClient()

    rows, version = iter_zotero_rows(client, include_attachments=True)

    assert version == 11
    assert [row["id"] for row in rows] == ["A1", "N1", "F1"]
    assert "Ada Lovelace" in rows[0]["text"]
    assert rows[1]["text"] == "Readable note"
    assert "Attachment body" in rows[2]["text"]


def test_iter_zotero_rows_uses_version_cursor_and_delete_tombstones(monkeypatch):
    monkeypatch.setattr(zotero_module, "ZOTERO_PAGE_SIZE", FakeClient.page_size)
    client = FakeClient()

    rows, version = iter_zotero_rows(client, since_version=7, include_notes=False)

    assert version == 12
    item_calls = [params for path, params in client.calls if path == "items"]
    assert item_calls[0]["since"] == 7
    assert item_calls[1]["start"] == FakeClient.page_size
    assert ("deleted", {"since": 7}) in client.calls
    assert [row["id"] for row in rows] == ["A1", "F1", "OLD"]
    assert rows[-1]["_deleted"] is True
