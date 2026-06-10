"""Unit tests for cognee.push() and the CMIF archive transport helpers.

All tests are pure: no databases, no LLM calls, no network.
"""

import asyncio
import importlib
import io
import tarfile

import pytest

from cognee.modules.migration.archive import (
    ARCHIVE_SUFFIX,
    find_archive_root,
    pack_archive,
    unpack_archive,
)
from cognee.modules.migration.cmif import (
    CMIFArchiveWriter,
    CMIFEntity,
    CMIFFact,
    read_archive,
    read_manifest,
)


def _write_sample_archive(directory):
    with CMIFArchiveWriter(directory, source_system="cognee") as writer:
        writer.write(CMIFEntity(external_system="cognee", external_id="e1", name="Alice"))
        writer.write(CMIFEntity(external_system="cognee", external_id="e2", name="Bob"))
        writer.write(
            CMIFFact(
                external_system="cognee",
                external_id="e1:knows:e2",
                subject_ref="e1",
                predicate="knows",
                object_ref="e2",
            )
        )


class TestArchiveTransport:
    def test_pack_unpack_round_trip(self, tmp_path):
        archive_dir = tmp_path / "cmif"
        _write_sample_archive(archive_dir)
        tar_path = tmp_path / f"sample{ARCHIVE_SUFFIX}"

        pack_archive(archive_dir, tar_path)
        assert tar_path.exists()

        extracted = tmp_path / "extracted"
        with open(tar_path, "rb") as archive_file:
            root = unpack_archive(archive_file, extracted)

        manifest = read_manifest(root)
        assert manifest is not None
        assert manifest.source_system == "cognee"
        records = list(read_archive(root))
        assert {record.kind for record in records} == {"entity", "fact"}
        assert len(records) == 3

    def test_unpack_rejects_path_traversal(self, tmp_path):
        malicious = io.BytesIO()
        with tarfile.open(fileobj=malicious, mode="w:gz") as tar:
            payload = b"evil"
            info = tarfile.TarInfo(name="../escape.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        malicious.seek(0)

        with pytest.raises(ValueError, match="Unsafe path"):
            unpack_archive(malicious, tmp_path / "out")

    def test_unpack_skips_symlinks(self, tmp_path):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            link = tarfile.TarInfo(name="link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            tar.addfile(link)
            payload = b"{}"
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        buffer.seek(0)

        root = unpack_archive(buffer, tmp_path / "out")
        assert not (root / "link").exists()
        assert (root / "manifest.json").exists()

    def test_find_archive_root_nested_one_level(self, tmp_path):
        nested = tmp_path / "wrapper" / "inner"
        _write_sample_archive(nested)
        assert find_archive_root(tmp_path / "wrapper") == nested

    def test_find_archive_root_missing_manifest(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="manifest.json"):
            find_archive_root(tmp_path / "empty")


class TestResolveClient:
    def _clear_sources(self, monkeypatch):
        from cognee.api.v1.serve import credentials as credentials_module
        from cognee.api.v1.serve import state

        monkeypatch.setattr(state, "_remote_client", None)
        monkeypatch.setattr(credentials_module, "load_credentials", lambda: None)
        monkeypatch.delenv("COGNEE_SERVICE_URL", raising=False)
        monkeypatch.delenv("COGNEE_API_KEY", raising=False)

    def test_explicit_url_wins(self, monkeypatch):
        from cognee.api.v1.push.push import _resolve_client

        self._clear_sources(monkeypatch)
        client, created = _resolve_client("https://explicit.example", "key1")
        assert created is True
        assert client.service_url == "https://explicit.example"
        assert client.api_key == "key1"

    def test_live_serve_connection_reused(self, monkeypatch):
        from cognee.api.v1.push.push import _resolve_client
        from cognee.api.v1.serve import state
        from cognee.api.v1.serve.cloud_client import CloudClient

        self._clear_sources(monkeypatch)
        live = CloudClient("https://live.example", "live-key")
        monkeypatch.setattr(state, "_remote_client", live)
        client, created = _resolve_client(None, None)
        assert created is False
        assert client is live

    def test_saved_credentials_used(self, monkeypatch):
        from cognee.api.v1.push.push import _resolve_client
        from cognee.api.v1.serve.credentials import CloudCredentials

        self._clear_sources(monkeypatch)
        from cognee.api.v1.serve import credentials as credentials_module

        creds = CloudCredentials(
            access_token="t", service_url="https://saved.example", api_key="saved-key"
        )
        monkeypatch.setattr(credentials_module, "load_credentials", lambda: creds)
        client, created = _resolve_client(None, None)
        assert created is True
        assert client.service_url == "https://saved.example"
        assert client.api_key == "saved-key"

    def test_env_fallback(self, monkeypatch):
        from cognee.api.v1.push.push import _resolve_client

        self._clear_sources(monkeypatch)
        monkeypatch.setenv("COGNEE_SERVICE_URL", "https://env.example")
        monkeypatch.setenv("COGNEE_API_KEY", "env-key")
        client, created = _resolve_client(None, None)
        assert created is True
        assert client.service_url == "https://env.example"
        assert client.api_key == "env-key"

    def test_no_connection_raises(self, monkeypatch):
        from cognee.api.v1.push.push import _resolve_client

        self._clear_sources(monkeypatch)
        with pytest.raises(RuntimeError, match="cognee-cli serve"):
            _resolve_client(None, None)


class TestPush:
    def test_push_uploads_cmif_archive(self, monkeypatch, tmp_path):
        push_module = importlib.import_module("cognee.api.v1.push.push")
        from cognee.modules.migration.export import ExportResult

        captured = {}

        async def fake_export(dataset, format, destination, user):
            _write_sample_archive(destination)
            return ExportResult(
                format=format,
                destination=str(destination),
                dataset_name="main_dataset",
                dataset_id="d-1",
                num_nodes=2,
                num_edges=1,
            )

        class FakeClient:
            service_url = "https://fake.example"
            closed = False

            async def remember(self, data, dataset_name, **kwargs):
                captured["dataset_name"] = dataset_name
                captured["kwargs"] = kwargs
                captured["payload"] = data.read()
                return {"status": "completed"}

            async def close(self):
                FakeClient.closed = True

        # The package re-exports the function, shadowing the submodule for
        # plain attribute access — resolve the real module via importlib.
        export_module = importlib.import_module("cognee.api.v1.export.export")
        monkeypatch.setattr(export_module, "export", fake_export)
        monkeypatch.setattr(
            push_module, "_resolve_client", lambda url, api_key: (FakeClient(), True)
        )

        result = asyncio.run(push_module.push("main_dataset", mode="hybrid"))

        assert result == {"status": "completed", "num_nodes": 2, "num_edges": 1}
        assert captured["dataset_name"] == "main_dataset"
        assert captured["kwargs"]["content_type"] == "cmif-archive"
        assert captured["kwargs"]["import_mode"] == "hybrid"
        assert FakeClient.closed is True

        # The uploaded payload is a valid, re-importable archive.
        root = unpack_archive(io.BytesIO(captured["payload"]), tmp_path / "out")
        assert len(list(read_archive(root))) == 3

    def test_push_target_dataset_override(self, monkeypatch):
        push_module = importlib.import_module("cognee.api.v1.push.push")
        from cognee.modules.migration.export import ExportResult

        captured = {}

        async def fake_export(dataset, format, destination, user):
            _write_sample_archive(destination)
            return ExportResult(
                format=format,
                destination=str(destination),
                dataset_name="local_name",
                dataset_id="d-1",
                num_nodes=0,
                num_edges=0,
            )

        class FakeClient:
            service_url = "https://fake.example"

            async def remember(self, data, dataset_name, **kwargs):
                captured["dataset_name"] = dataset_name
                return {"status": "completed"}

            async def close(self):
                pass

        # The package re-exports the function, shadowing the submodule for
        # plain attribute access — resolve the real module via importlib.
        export_module = importlib.import_module("cognee.api.v1.export.export")
        monkeypatch.setattr(export_module, "export", fake_export)
        monkeypatch.setattr(
            push_module, "_resolve_client", lambda url, api_key: (FakeClient(), True)
        )

        asyncio.run(push_module.push("local_name", target_dataset="remote_name"))
        assert captured["dataset_name"] == "remote_name"

    def test_push_invalid_mode(self):
        from cognee.api.v1.push.push import push

        with pytest.raises(ValueError, match="Unknown push mode"):
            asyncio.run(push("main_dataset", mode="bogus"))
