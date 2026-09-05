from __future__ import annotations

import json
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR
except ImportError:
    DOCUMENT_SOURCE_ATTR = "cognee_document_source"


ZOTERO_PAGE_SIZE = 100


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def _html_text(value: str | None) -> str:
    parser = _HTMLText()
    parser.feed(value or "")
    return unescape(" ".join(parser.parts))


@dataclass
class ZoteroClient:
    api_key: str
    user_id: str | None = None
    group_id: str | None = None
    base_url: str = "https://api.zotero.org"

    def __post_init__(self) -> None:
        if bool(self.user_id) == bool(self.group_id):
            raise ValueError("Pass exactly one of user_id or group_id.")

    @property
    def library_path(self) -> str:
        kind, value = ("users", self.user_id) if self.user_id else ("groups", self.group_id)
        return f"{kind}/{value}"

    def get_json(self, path: str, params: dict | None = None) -> tuple[object, int | None]:
        url = f"{self.base_url.rstrip('/')}/{self.library_path}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode({k: v for k, v in params.items() if v is not None})}"
        request = Request(url, headers={"Zotero-API-Key": self.api_key})
        with urlopen(request, timeout=30) as response:
            version = response.headers.get("Last-Modified-Version")
            return json.loads(response.read().decode("utf-8")), int(version) if version else None


def iter_zotero_rows(
    client: ZoteroClient,
    *,
    since_version: int | None = None,
    include_references: bool = True,
    include_notes: bool = True,
    include_attachments: bool = True,
) -> tuple[list[dict], int | None]:
    rows: list[dict] = []
    version = None
    start = 0
    while True:
        changed, page_version = client.get_json(
            "items",
            {
                "format": "json",
                "include": "data",
                "since": since_version,
                "limit": ZOTERO_PAGE_SIZE,
                "start": start,
            },
        )
        if page_version is not None:
            version = max(version or page_version, page_version)
        items = changed if isinstance(changed, list) else []
        for item in items:
            data = item.get("data", {})
            item_type = data.get("itemType", "reference")
            kind = (
                "note"
                if item_type == "note"
                else "attachment"
                if item_type == "attachment"
                else "reference"
            )
            if kind == "reference" and not include_references:
                continue
            if kind == "note" and not include_notes:
                continue
            if kind == "attachment" and not include_attachments:
                continue
            rows.append(_row_for_item(data, kind, client if include_attachments else None))
        if len(items) < ZOTERO_PAGE_SIZE:
            break
        start += ZOTERO_PAGE_SIZE

    if since_version is not None:
        deleted, deleted_version = client.get_json("deleted", {"since": since_version})
        version = max(v for v in (version, deleted_version) if v is not None) if any(
            v is not None for v in (version, deleted_version)
        ) else version
        deleted_keys = (deleted or {}).get("items", []) if isinstance(deleted, dict) else []
        for key in deleted_keys:
            rows.append({"id": key, "kind": "deleted", "text": "", "version": version, "_deleted": True})

    return rows, version


def _row_for_item(data: dict, kind: str, client: ZoteroClient | None) -> dict:
    key = data["key"]
    title = data.get("title") or data.get("filename") or key
    chunks = [title if title != key else None, data.get("abstractNote"), _html_text(data.get("note"))]
    if kind == "attachment" and client is not None:
        fulltext, _ = client.get_json(f"items/{key}/fulltext")
        if isinstance(fulltext, dict):
            chunks.append(fulltext.get("content"))
    creators = [
        " ".join(part for part in (c.get("firstName"), c.get("lastName")) if part)
        for c in data.get("creators", [])
    ]
    if creators:
        chunks.append("Authors: " + ", ".join(creators))
    return {
        "id": key,
        "kind": kind,
        "title": title,
        "text": "\n".join(str(chunk) for chunk in chunks if chunk),
        "version": data.get("version"),
        "_deleted": False,
    }


def zotero_source(
    *,
    api_key: str,
    user_id: str | None = None,
    group_id: str | None = None,
    include_references: bool = True,
    include_notes: bool = True,
    include_attachments: bool = True,
    base_url: str = "https://api.zotero.org",
):
    import dlt

    client = ZoteroClient(api_key=api_key, user_id=user_id, group_id=group_id, base_url=base_url)

    @dlt.resource(
        name="zotero_items",
        primary_key="id",
        write_disposition="merge",
        columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
    )
    def zotero_items() -> Iterable[dict]:
        state = dlt.current.source_state()
        rows, version = iter_zotero_rows(
            client,
            since_version=state.get("version"),
            include_references=include_references,
            include_notes=include_notes,
            include_attachments=include_attachments,
        )
        yield from rows
        if version is not None:
            state["version"] = version

    setattr(zotero_items, DOCUMENT_SOURCE_ATTR, "zotero")
    return zotero_items
