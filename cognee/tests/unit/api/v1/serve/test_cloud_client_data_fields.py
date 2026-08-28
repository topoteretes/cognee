"""CloudClient must resolve ``data`` the way local ingestion does.

Local ``add``/``remember`` treat ``file://`` URIs and existing paths as
files, expand directories, and honour ``DataItem`` labels/metadata. The
remote client used to upload every string as raw text (so a file path was
ingested as the literal path string under a ``.txt`` name — losing the
extension the server uses for loader routing) and dropped ``DataItem``
inputs entirely.
"""

import asyncio
import io
import json

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.tasks.ingestion.data_item import DataItem


class FakeResponse:
    status = 200

    async def json(self):
        return {}

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    closed = False

    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return FakeResponse()


def make_client():
    client = CloudClient("http://cloud.example", "ck_test")
    session = FakeSession()
    client._session = session
    return client, session


def form_parts(session):
    """Return the multipart parts of the last request as (name, filename, content_type, value)."""
    form = session.calls[-1]["data"]
    parts = []
    for type_options, headers, value in form._fields:
        name = type_options["name"]
        filename = type_options.get("filename")
        parts.append((name, filename, headers.get("Content-Type"), value))
    return parts


def data_parts(session):
    return [part for part in form_parts(session) if part[0] == "data"]


def field_value(session, name):
    matches = [part for part in form_parts(session) if part[0] == name]
    return matches[0][3] if matches else None


# ----- file paths -----


def test_existing_path_uploads_file_under_real_basename(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("def f():\n    return 1\n")
    client, session = make_client()

    asyncio.run(client.remember(str(source)))

    (part,) = data_parts(session)
    assert part[1] == "module.py"
    assert part[3].closed  # opened by the client, closed after the request


def test_file_uri_uploads_file(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# notes")
    client, session = make_client()

    asyncio.run(client.add(source.as_uri()))

    (part,) = data_parts(session)
    assert part[1] == "notes.md"


def test_missing_file_uri_raises(tmp_path):
    client, _ = make_client()
    with pytest.raises(FileNotFoundError):
        asyncio.run(client.remember((tmp_path / "absent.txt").as_uri()))


def test_directory_expands_to_its_files_recursively(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    client, session = make_client()

    asyncio.run(client.remember(str(tmp_path)))

    assert sorted(part[1] for part in data_parts(session)) == ["a.txt", "b.txt"]


def test_path_object_is_accepted(tmp_path):
    source = tmp_path / "doc.txt"
    source.write_text("doc")
    client, session = make_client()

    asyncio.run(client.add(source))

    assert data_parts(session)[0][1] == "doc.txt"


def test_non_path_string_is_uploaded_as_text():
    client, session = make_client()

    asyncio.run(client.remember("/remember to call Bob"))

    (part,) = data_parts(session)
    assert part[1].startswith("text_") and part[1].endswith(".txt")
    assert part[2] == "text/plain"
    assert part[3].getvalue() == b"/remember to call Bob"


def test_remote_url_is_rejected_with_actionable_error():
    client, _ = make_client()
    with pytest.raises(ValueError, match="https:// sources cannot be ingested"):
        asyncio.run(client.remember("https://example.com/page"))
    with pytest.raises(ValueError, match="s3:// sources"):
        asyncio.run(client.add("s3://bucket/key.pdf"))


def test_mixed_list_of_text_and_files(tmp_path):
    source = tmp_path / "spec.pdf"
    source.write_bytes(b"%PDF-1.4")
    client, session = make_client()

    asyncio.run(client.remember(["plain note", str(source)]))

    names = [part[1] for part in data_parts(session)]
    assert names[0].startswith("text_")
    assert names[1] == "spec.pdf"


def test_file_like_objects_keep_basename_only():
    handle = io.BytesIO(b"x")
    handle.name = "/tmp/somewhere/report.csv"
    client, session = make_client()

    asyncio.run(client.add(handle))

    assert data_parts(session)[0][1] == "report.csv"


# ----- DataItem -----


def test_data_item_text_is_uploaded_with_label_and_metadata():
    client, session = make_client()

    asyncio.run(
        client.remember(
            [
                DataItem(data="finance note", label="finance", external_metadata={"src": "crm"}),
                DataItem(data="untagged note"),
            ]
        )
    )

    assert len(data_parts(session)) == 2
    assert json.loads(field_value(session, "labels")) == ["finance", ""]
    assert json.loads(field_value(session, "external_metadata")) == [{"src": "crm"}, None]


def test_data_item_file_path_is_uploaded(tmp_path):
    source = tmp_path / "ledger.csv"
    source.write_text("a,b\n1,2\n")
    client, session = make_client()

    asyncio.run(client.add(DataItem(data=str(source), label="ledger")))

    (part,) = data_parts(session)
    assert part[1] == "ledger.csv"
    assert json.loads(field_value(session, "labels")) == ["ledger"]


def test_no_labels_field_when_no_item_carries_one():
    client, session = make_client()

    asyncio.run(client.remember([DataItem(data="a"), "b"]))

    assert field_value(session, "labels") is None
    assert field_value(session, "external_metadata") is None


# ----- search session_id -----


def test_search_forwards_session_id():
    client, session = make_client()

    asyncio.run(client.search("q", session_id="oc_session"))

    assert session.calls[-1]["json"]["sessionId"] == "oc_session"
