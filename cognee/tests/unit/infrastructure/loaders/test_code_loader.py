"""Code files are recognized through the loader system.

The code loader claiming a file at add time IS the recognition that routes it
down the cognify CODE route (ingest_data tags system_metadata.source="code"
when the selected loader is the code loader). These tests lock in the
selection seam: code extensions go to code_loader, and — because code files
content-sniff as plain text — code_loader must precede text_loader in the
default priority, without stealing genuine text files.
"""

import pytest

from cognee.infrastructure.loaders.core.code_loader import (
    SUPPORTED_CODE_EXTENSIONS,
    CodeLoader,
)
from cognee.infrastructure.loaders.create_loader_engine import create_loader_engine


@pytest.fixture
def engine():
    return create_loader_engine()


def test_code_file_selects_code_loader(engine, tmp_path):
    code_file = tmp_path / "payments.py"
    code_file.write_text("def hello():\n    pass\n")

    loader = engine.get_loader(str(code_file), preferred_loaders=None)

    assert loader.loader_name == "code_loader"


def test_text_file_still_selects_text_loader(engine, tmp_path):
    """code_loader precedes text_loader in priority but must not claim text."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("plain text, not code")

    loader = engine.get_loader(str(text_file), preferred_loaders=None)

    assert loader.loader_name == "text_loader"


def test_preferred_loader_overrides_code_recognition(engine, tmp_path):
    """A preferred_loaders override away from code_loader must win — the same
    override also opts the file out of the CODE cognify route."""
    code_file = tmp_path / "payments.py"
    code_file.write_text("def hello():\n    pass\n")

    loader = engine.get_loader(str(code_file), preferred_loaders={"text_loader": {}})

    assert loader.loader_name == "text_loader"


def test_can_handle_is_extension_only():
    """Content sniffing reports code as txt/text-plain; the code loader must
    key on the file-name extension alone and never claim the sniffed txt."""
    loader = CodeLoader()

    assert loader.can_handle(extension="py", mime_type="text/plain")
    assert loader.can_handle(extension="GO", mime_type=None)
    assert not loader.can_handle(extension="txt", mime_type="text/plain")


def test_supported_extensions_exclude_generic_formats():
    """yml/json/etc. double as ordinary documents; claiming them would hijack
    non-code files into the code route at add time."""
    for generic in ("yml", "yaml", "json", "xml", "md", "txt", "html"):
        assert generic not in SUPPORTED_CODE_EXTENSIONS


@pytest.mark.asyncio
async def test_load_keeps_code_extension_in_storage_name(tmp_path, monkeypatch):
    """The storage file must keep the real extension: content sniffing calls
    code "txt", so the raw_data_location basename is the only place the
    language extension survives for enola."""
    import cognee.infrastructure.loaders.core.code_loader as code_loader_module

    code_file = tmp_path / "payments.py"
    code_file.write_text("def hello():\n    pass\n")
    stored = {}

    class _FakeStorage:
        async def store(self, name, content):
            stored["name"], stored["content"] = name, content
            return f"/storage/{name}"

    monkeypatch.setattr(code_loader_module, "get_file_storage", lambda root: _FakeStorage())
    monkeypatch.setattr(
        code_loader_module, "get_storage_config", lambda: {"data_root_directory": "/storage"}
    )

    result = await CodeLoader().load(str(code_file))

    assert stored["name"].startswith("code_") and stored["name"].endswith(".py")
    assert stored["content"] == "def hello():\n    pass\n"
    # The LoaderResult route stamp IS the recognition ingest_data writes.
    assert result.file_path == f"/storage/{stored['name']}"
    assert result.system_metadata == {"source": "code"}


@pytest.mark.asyncio
async def test_s3_code_file_selects_code_loader(monkeypatch):
    """S3 ingestion must recognize code files too: the S3 branch downloads the
    object into a temp file that KEEPS the original suffix, and the loader is
    selected from that temp file — so a .py object reaches code_loader exactly
    like a local path. Locks in the suffix-preserving temp-file contract."""
    import cognee.infrastructure.loaders.core.code_loader as code_loader_module
    import cognee.tasks.ingestion.data_item_to_text_file as ditf

    async def fake_pull_from_s3(file_path, destination_file):
        destination_file.write(b"def hello():\n    pass\n")

    stored = {}

    class _FakeStorage:
        async def store(self, name, content):
            stored["name"] = name
            return f"/storage/{name}"

    monkeypatch.setattr(ditf, "pull_from_s3", fake_pull_from_s3)
    monkeypatch.setattr(code_loader_module, "get_file_storage", lambda root: _FakeStorage())
    monkeypatch.setattr(
        code_loader_module, "get_storage_config", lambda: {"data_root_directory": "/storage"}
    )

    loader_result, loader = await ditf.data_item_to_text_file("s3://bucket/src/payments.py")

    assert loader.loader_name == "code_loader"
    assert stored["name"].startswith("code_") and stored["name"].endswith(".py")
    assert loader_result.file_path == f"/storage/{stored['name']}"
    assert loader_result.system_metadata == {"source": "code"}


def test_staging_never_overwrites_a_file_that_is_its_own_marker(tmp_path):
    """A user-added Package.swift IS the swift detection marker's filename;
    staging must keep the user's content, not clobber it with the stub."""
    from types import SimpleNamespace
    from uuid import uuid4

    from cognee.tasks.code_graph.code_files import _stage_code_file

    user_content = 'let package = Package(name: "the-users-real-package")\n'
    data_item = SimpleNamespace(id=uuid4(), original_data_location="/repo/Package.swift")

    repo_dir = _stage_code_file(data_item, str(tmp_path), user_content)

    assert (repo_dir / "Package.swift").read_text() == user_content


def test_staging_writes_markers_next_to_ordinary_code_files(tmp_path):
    from types import SimpleNamespace
    from uuid import uuid4

    from cognee.tasks.code_graph.code_files import _stage_code_file

    data_item = SimpleNamespace(id=uuid4(), original_data_location="/repo/payments.py")

    repo_dir = _stage_code_file(data_item, str(tmp_path), "def hello():\n    pass\n")

    assert (repo_dir / "payments.py").read_text() == "def hello():\n    pass\n"
    assert (repo_dir / "requirements.txt").exists()


def test_code_route_tasks_bind_the_index_vectors_choice():
    """cognify passes CognifyConfig.code_route_index_vectors (default ON) into
    the route's task list; the task must carry it to the enola load stage."""
    from cognee.tasks.code_graph.code_files import get_code_file_tasks

    (task_on,) = get_code_file_tasks(index_vectors=True)
    (task_off,) = get_code_file_tasks()

    assert task_on.default_params["kwargs"]["index_vectors"] is True
    assert task_off.default_params["kwargs"]["index_vectors"] is False
