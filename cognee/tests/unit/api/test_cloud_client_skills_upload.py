"""CloudClient.remember(content_type='skills') reads local SKILL.md files
and uploads their contents — not the path string.

Before the fix, a remote ``cognee.remember("./skills", content_type="skills")``
sent the literal string ``"./skills"`` (10 bytes) as the form body. The pod
then handed an UploadFile containing that string to ``add_skills`` which
rejected it with PermissionError because no such path exists pod-side.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from cognee.api.v1.serve.cloud_client import CloudClient


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"status": "completed"}

    async def text(self):
        return ""


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in that captures the POST body."""

    def __init__(self):
        self.last_form = None
        self.closed = False

    def post(self, _url, data=None, **kwargs):
        self.last_form = data
        return _FakeResponse()

    async def close(self):
        self.closed = True


def _form_field_dump(form):
    """Extract (name, filename, body_bytes) tuples from an aiohttp.FormData."""
    out = []
    # FormData stores fields in ._fields (list of (type_options, headers, value))
    for type_options, _headers, value in form._fields:
        name = type_options.get("name")
        filename = type_options.get("filename")
        if hasattr(value, "read"):
            body = value.read()
        elif isinstance(value, (bytes, bytearray)):
            body = bytes(value)
        else:
            body = str(value).encode("utf-8")
        out.append((name, filename, body))
    return out


def test_remember_skills_with_file_path_uploads_bytes(tmp_path):
    skill = tmp_path / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Demo Skill\n\nDoes the thing.\n", encoding="utf-8")

    client = CloudClient(service_url="https://example.test", api_key="k")
    fake_session = _FakeSession()

    async def run():
        with patch.object(client, "_get_session", return_value=fake_session):
            await client.remember(
                str(tmp_path),
                dataset_name="ds",
                content_type="skills",
            )

    asyncio.run(run())

    fields = _form_field_dump(fake_session.last_form)
    skill_fields = [(name, fn, body) for name, fn, body in fields if name == "data"]
    assert skill_fields, "expected at least one 'data' field"
    # filename preserves the relative layout under the source dir
    assert skill_fields[0][1] == "demo/SKILL.md", skill_fields[0][1]
    # body is the FILE contents, not the path string
    assert skill_fields[0][2].decode("utf-8").startswith("# Demo Skill"), skill_fields[0][2]
    # content_type=skills made it onto the form
    ct = [(name, body.decode("utf-8")) for name, _fn, body in fields if name == "content_type"]
    assert ct == [("content_type", "skills")], ct


def test_remember_skills_rejects_missing_source(tmp_path):
    client = CloudClient(service_url="https://example.test", api_key="k")
    fake_session = _FakeSession()

    async def run():
        with patch.object(client, "_get_session", return_value=fake_session):
            await client.remember(
                str(tmp_path / "nope"),
                dataset_name="ds",
                content_type="skills",
            )

    try:
        asyncio.run(run())
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing skills source")


def test_remember_skills_rejects_empty_folder(tmp_path):
    client = CloudClient(service_url="https://example.test", api_key="k")
    fake_session = _FakeSession()

    async def run():
        with patch.object(client, "_get_session", return_value=fake_session):
            await client.remember(
                str(tmp_path),
                dataset_name="ds",
                content_type="skills",
            )

    try:
        asyncio.run(run())
    except ValueError as e:
        assert "No SKILL.md files" in str(e)
        return
    raise AssertionError("expected ValueError for empty skills folder")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        test_remember_skills_with_file_path_uploads_bytes(p)
        # Fresh dir for the missing-source case
    with tempfile.TemporaryDirectory() as tmp:
        test_remember_skills_rejects_missing_source(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_remember_skills_rejects_empty_folder(Path(tmp))
    print("OK")
