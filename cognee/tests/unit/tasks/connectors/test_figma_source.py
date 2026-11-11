"""Unit tests for figma_source: verifying the 5 core issue requirements.

1. Authenticate using the Figma API.
2. Retrieve Figma files, pages, and text elements.
3. Extract comments and design metadata.
4. Preserve file and page relationships.
5. Support incremental synchronization.
"""

import pytest

# pyrefly: ignore[missing-import]
from fake_figma import FakeFigma

from cognee.tasks.ingestion.connectors import figma_source
from cognee.tasks.ingestion.connectors.figma import _clean_file_keys, _iter_rows, new_stats
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
    STRUCTURE_COLUMN,
)


def _sync(fake: FakeFigma, state: dict, file_keys: list[str], **kwargs) -> tuple[list[dict], dict]:
    stats = new_stats()
    rows = list(_iter_rows(fake, state, stats, file_keys, **kwargs))
    return rows, stats


# ---------------------------------------------------------------------------
# Requirement 1: Authentication & Parameter Validation
# ---------------------------------------------------------------------------
def test_authentication_and_parameter_validation():
    with pytest.raises(ValueError, match="needs a service or an access_token"):
        figma_source(resource_name="test_figma", file_keys=["abc123"])

    with pytest.raises(ValueError, match="resource_name, one per Figma connection"):
        figma_source(resource_name="", file_keys=["abc123"], access_token="token")

    with pytest.raises(ValueError, match="file_keys must be a non-empty list or tuple"):
        figma_source(resource_name="test_figma", file_keys=[], access_token="token")

    with pytest.raises(ValueError, match="must contain non-empty strings"):
        figma_source(resource_name="test_figma", file_keys=[""], access_token="token")

    clean = _clean_file_keys(["  key2  ", "key1", "key2"])
    assert clean == ["key1", "key2"]

    # URL extraction
    url_clean = _clean_file_keys(["https://www.figma.com/design/abc123XYZ/App-Design"])
    assert url_clean == ["abc123XYZ"]
    proto_clean = _clean_file_keys(
        ["https://www.figma.com/proto/protoKey987XYZ/App-Prototype?node-id=0-1"]
    )
    assert proto_clean == ["protoKey987XYZ"]


def test_dlt_resource_metadata_attributes():
    fake = FakeFigma()
    resource = figma_source(resource_name="my_figma", file_keys=["file1"], service=fake)

    assert getattr(resource, DOCUMENT_SOURCE_ATTR) == "figma"
    assert getattr(resource, PIPELINE_SCOPE_ATTR) == "my_figma"
    assert hasattr(resource, "cognee_sync_stats")


# ---------------------------------------------------------------------------
# Requirement 2 & 3: File, Page, Text, Comments & Design Metadata Retrieval
# ---------------------------------------------------------------------------
def test_retrieve_files_pages_text_comments_and_metadata():
    fake = FakeFigma()
    fake.add_file(
        "file1",
        name="Mobile App Design",
        version="v1.0.4",
        last_modified="2026-10-31T10:00:00Z",
        components={"comp_btn": {"name": "PrimaryButton", "description": "Standard action button"}},
        pages=[
            {
                "id": "10:1",
                "name": "Auth Flow",
                "type": "CANVAS",
                "children": [
                    {
                        "id": "10:2",
                        "name": "Login Screen",
                        "type": "FRAME",
                        "children": [
                            {
                                "id": "10:3",
                                "name": "Heading",
                                "type": "TEXT",
                                "characters": "Sign In to Your Account",
                            },
                            {
                                "id": "10:4",
                                "name": "PrimaryButton",
                                "type": "COMPONENT",
                            },
                        ],
                    }
                ],
            }
        ],
    )
    fake.add_comment("file1", "c1", "Overall look is great", node_id=None, user="Alice")
    fake.add_comment("file1", "c2", "Make this title bolder", node_id="10:3", user="Bob")

    state: dict = {}
    rows, stats = _sync(fake, state, ["file1"])

    assert len(rows) == 2  # File overview document + 1 Canvas/Page document
    by_id = {r["id"]: r for r in rows}

    # Verify File Document
    file_doc = by_id["file:file1"]
    assert file_doc["title"] == "Mobile App Design"
    assert file_doc["url"] == "https://www.figma.com/design/file1"
    assert not file_doc["_deleted"]
    assert file_doc[NODE_SET_COLUMN] == ["figma:file1"]
    assert "Version: v1.0.4" in file_doc["content"]
    assert "Last Modified: 2026-10-31T10:00:00Z" in file_doc["content"]
    assert "PrimaryButton: Standard action button" in file_doc["content"]
    assert "- Alice: Overall look is great" in file_doc["content"]

    # Verify Page Document & Text Extraction
    page_doc = by_id["page:file1:10:1"]
    assert page_doc["title"] == "Mobile App Design / Auth Flow"
    assert page_doc["url"] == "https://www.figma.com/design/file1?node-id=10:1"
    assert not page_doc["_deleted"]
    assert page_doc[NODE_SET_COLUMN] == ["figma:file1"]
    assert "Page: Auth Flow" in page_doc["content"]
    assert "### Frame: Login Screen" in page_doc["content"]
    assert "[Login Screen] Sign In to Your Account" in page_doc["content"]
    assert "Component: PrimaryButton" in page_doc["content"]
    assert "Comments on Heading:" in page_doc["content"]
    assert "- Bob: Make this title bolder" in page_doc["content"]

    assert stats["scanned"] == 2
    assert stats["skipped"] == 0
    assert stats["deleted"] == 0


# ---------------------------------------------------------------------------
# Requirement 4: Relationship Preservation (STRUCTURE_COLUMN)
# ---------------------------------------------------------------------------
def test_preserves_file_and_page_relationships():
    fake = FakeFigma()
    fake.add_file("file1", name="Design System")

    state: dict = {}
    rows, _ = _sync(fake, state, ["file1"])
    by_id = {r["id"]: r for r in rows}

    file_doc = by_id["file:file1"]
    page_doc = by_id["page:file1:0:1"]

    # File is top-level (empty ancestors)
    assert file_doc[STRUCTURE_COLUMN] == {"ancestors": []}

    # Page points to file document as parent document
    assert page_doc[STRUCTURE_COLUMN] == {
        "ancestors": [
            {
                "kind": "file",
                "id": "file:file1",
                "name": "Design System",
                "document": True,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Requirement 5: Incremental Synchronization & Deletions
# ---------------------------------------------------------------------------
def test_incremental_sync_skips_unmodified_files():
    fake = FakeFigma()
    fake.add_file("file1", name="Design System", last_modified="2026-10-31T08:00:00Z")

    state: dict = {}
    rows1, stats1 = _sync(fake, state, ["file1"])
    assert len(rows1) == 2
    assert stats1["scanned"] == 2
    assert stats1["skipped"] == 0

    # Second sync: timestamp unchanged -> skip processing
    rows2, stats2 = _sync(fake, state, ["file1"])
    assert len(rows2) == 0
    assert stats2["skipped"] == 1
    assert stats2["scanned"] == 0


def test_incremental_sync_updates_when_last_modified_advances():
    fake = FakeFigma()
    fake.add_file("file1", name="Design System", last_modified="2026-10-31T08:00:00Z")

    state: dict = {}
    _sync(fake, state, ["file1"])

    # File modified in Figma
    fake.files["file1"]["lastModified"] = "2026-10-31T09:00:00Z"
    fake.files["file1"]["document"]["children"][0]["name"] = "Updated Page"

    rows, stats = _sync(fake, state, ["file1"])
    assert len(rows) == 2
    assert stats["scanned"] == 2
    assert stats["skipped"] == 0
    page_doc = next(r for r in rows if r["id"].startswith("page:"))
    assert "Page: Updated Page" in page_doc["content"]


def test_incremental_sync_emits_tombstone_when_page_deleted():
    fake = FakeFigma()
    fake.add_file(
        "file1",
        pages=[
            {"id": "0:1", "name": "Page 1", "type": "CANVAS", "children": []},
            {"id": "0:2", "name": "Page 2", "type": "CANVAS", "children": []},
        ],
        last_modified="2026-10-31T08:00:00Z",
    )

    state: dict = {}
    rows1, _ = _sync(fake, state, ["file1"])
    assert {r["id"] for r in rows1} == {"file:file1", "page:file1:0:1", "page:file1:0:2"}

    # Page 2 is deleted in Figma
    fake.files["file1"]["lastModified"] = "2026-10-31T09:00:00Z"
    fake.files["file1"]["document"]["children"] = [
        {"id": "0:1", "name": "Page 1", "type": "CANVAS", "children": []}
    ]

    rows2, stats2 = _sync(fake, state, ["file1"])
    assert stats2["deleted"] == 1
    # Check that tombstone for page 2 was emitted
    tombstones = [r for r in rows2 if r["_deleted"]]
    assert len(tombstones) == 1
    assert tombstones[0]["id"] == "page:file1:0:2"


def test_deselecting_file_emits_tombstones_for_all_its_documents():
    fake = FakeFigma()
    fake.add_file("file1")
    fake.add_file("file2")

    state: dict = {}
    _sync(fake, state, ["file1", "file2"])

    # Deselect file2
    rows, stats = _sync(fake, state, ["file1"])
    deleted_ids = {r["id"] for r in rows if r["_deleted"]}
    assert "file:file2" in deleted_ids
    assert "page:file2:0:1" in deleted_ids
    assert stats["deleted"] == 2
    assert "file2" not in state["files"]
