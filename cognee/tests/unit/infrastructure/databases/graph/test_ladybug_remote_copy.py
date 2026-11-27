"""A Ladybug graph on an object store (s3://, az://) is worked on in a local copy, which
push_to_remote uploads: in subprocess mode (the default) as in in-process mode.

Regression: subprocess mode handed the worker the remote URL as a database path, so the
graph landed in a local directory named ``s3:`` / ``az:`` and never reached the store.
"""

import pytest

from cognee.infrastructure.databases.graph.ladybug import adapter as adapter_module
from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter


class _RemoteGraphStore:
    """Stands in for the object-store backend: nothing is stored yet; uploads are recorded."""

    def __init__(self):
        self.uploads = []

    async def download_to(self, local_path):
        raise FileNotFoundError(local_path)

    async def upload_from(self, local_path):
        self.uploads.append(local_path)


@pytest.mark.parametrize("scheme, backend", [("s3", "s3"), ("az", "azure")])
@pytest.mark.asyncio
async def test_subprocess_mode_works_on_a_local_copy_of_a_remote_graph(
    monkeypatch, tmp_path, scheme, backend
):
    store = _RemoteGraphStore()
    monkeypatch.setattr(adapter_module, "get_remote_file_storage", lambda path: store)
    monkeypatch.setenv("STORAGE_BACKEND", backend)
    monkeypatch.chdir(tmp_path)  # a URL opened as a local path would land here

    adapter = LadybugAdapter.create_subprocess(db_path=f"{scheme}://box/databases/graph.lbug")
    try:
        assert not (tmp_path / f"{scheme}:").exists(), "the worker opened the URL as a path"
        assert adapter.needs_push_to_remote()
        await adapter.push_to_remote()
        assert store.uploads == [adapter.temp_graph_file]
    finally:
        await adapter.close()
