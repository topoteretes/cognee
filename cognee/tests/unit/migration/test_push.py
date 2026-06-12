"""Unit tests for cognee.push() and the COGX archive transport helpers.

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
from cognee.modules.migration.cogx import (
    COGXArchiveWriter,
    COGXEntity,
    COGXFact,
    read_archive,
    read_manifest,
)


def _make_tarball(files):
    """Build an in-memory ``.tar.gz`` from ``{name: payload_bytes}``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    buffer.seek(0)
    return buffer


def _write_sample_archive(directory):
    with COGXArchiveWriter(directory, source_system="cognee") as writer:
        writer.write(COGXEntity(external_system="cognee", external_id="e1", name="Alice"))
        writer.write(COGXEntity(external_system="cognee", external_id="e2", name="Bob"))
        writer.write(
            COGXFact(
                external_system="cognee",
                external_id="e1:knows:e2",
                subject_ref="e1",
                predicate="knows",
                object_ref="e2",
            )
        )


class TestArchiveTransport:
    def test_pack_unpack_round_trip(self, tmp_path):
        archive_dir = tmp_path / "cogx"
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


class TestArchiveExtractionLimits:
    """Decompression-bomb caps on ``unpack_archive`` (H4)."""

    def test_rejects_too_many_members(self, tmp_path):
        tarball = _make_tarball({f"file{i}.jsonl": b"{}" for i in range(4)})
        with pytest.raises(ValueError, match="more than 3 members"):
            unpack_archive(tarball, tmp_path / "out", max_members=3)

    def test_rejects_oversized_member(self, tmp_path):
        tarball = _make_tarball({"big.jsonl": b"x" * 100})
        with pytest.raises(ValueError, match="per-member limit"):
            unpack_archive(tarball, tmp_path / "out", max_member_bytes=10)

    def test_rejects_oversized_total(self, tmp_path):
        # Each member fits the per-member cap; together they exceed the total.
        tarball = _make_tarball({"a.jsonl": b"x" * 60, "b.jsonl": b"y" * 60})
        with pytest.raises(ValueError, match="in total"):
            unpack_archive(tarball, tmp_path / "out", max_member_bytes=80, max_total_bytes=100)

    def test_aborted_extraction_cleans_up(self, tmp_path):
        tarball = _make_tarball({"a.jsonl": b"x" * 60, "b.jsonl": b"y" * 60})
        destination = tmp_path / "out"
        with pytest.raises(ValueError):
            unpack_archive(tarball, destination, max_member_bytes=80, max_total_bytes=100)
        # The first member was extracted before the abort; it must be removed.
        assert list(destination.iterdir()) == []

    def test_limits_within_bounds_extracts_normally(self, tmp_path):
        archive_dir = tmp_path / "cogx"
        _write_sample_archive(archive_dir)
        tar_path = tmp_path / f"sample{ARCHIVE_SUFFIX}"
        pack_archive(archive_dir, tar_path)

        with open(tar_path, "rb") as archive_file:
            root = unpack_archive(
                archive_file,
                tmp_path / "out",
                max_members=10,
                max_member_bytes=1024 * 1024,
                max_total_bytes=1024 * 1024,
            )
        assert len(list(read_archive(root))) == 3


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

    def test_env_beats_saved_credentials(self, monkeypatch):
        """Precedence matches serve(): COGNEE_SERVICE_URL wins over saved creds."""
        from cognee.api.v1.push.push import _resolve_client
        from cognee.api.v1.serve import credentials as credentials_module
        from cognee.api.v1.serve.credentials import CloudCredentials

        self._clear_sources(monkeypatch)
        creds = CloudCredentials(
            access_token="t", service_url="https://saved.example", api_key="saved-key"
        )
        monkeypatch.setattr(credentials_module, "load_credentials", lambda: creds)
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


# A remote response whose items prove the server ran the migration import path.
_MIGRATION_RESPONSE = {
    "status": "completed",
    "items": [{"kind": "migration_import", "entities": 2, "facts": 1}],
}


def _fake_export(dataset_name="main_dataset", num_nodes=2, num_edges=1):
    from cognee.modules.migration.export import ExportResult

    async def fake_export(dataset, format, destination, user):
        _write_sample_archive(destination)
        return ExportResult(
            format=format,
            destination=str(destination),
            dataset_name=dataset_name,
            dataset_id="d-1",
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

    return fake_export


class TestPush:
    def _patch(self, monkeypatch, fake_export, fake_client):
        push_module = importlib.import_module("cognee.api.v1.push.push")
        # The package re-exports the function, shadowing the submodule for
        # plain attribute access — resolve the real module via importlib.
        export_module = importlib.import_module("cognee.api.v1.export.export")
        monkeypatch.setattr(export_module, "export", fake_export)
        monkeypatch.setattr(
            push_module, "_resolve_client", lambda url, api_key: (fake_client, True)
        )
        return push_module

    def test_push_uploads_cogx_archive(self, monkeypatch, tmp_path):
        captured = {}

        class FakeClient:
            service_url = "https://fake.example"
            closed = False

            async def remember(self, data, dataset_name, **kwargs):
                captured["dataset_name"] = dataset_name
                captured["kwargs"] = kwargs
                captured["payload"] = data.read()
                return dict(_MIGRATION_RESPONSE)

            async def close(self):
                FakeClient.closed = True

        push_module = self._patch(monkeypatch, _fake_export(), FakeClient())

        result = asyncio.run(push_module.push("main_dataset", mode="hybrid"))

        assert result.status == "completed"
        assert result.dataset_name == "main_dataset"
        assert result.target_dataset == "main_dataset"
        assert result.num_nodes == 2
        assert result.num_edges == 1
        assert result.remote_response["status"] == "completed"
        assert captured["dataset_name"] == "main_dataset"
        assert captured["kwargs"]["content_type"] == "cogx-archive"
        assert captured["kwargs"]["import_mode"] == "hybrid"
        assert FakeClient.closed is True

        # The uploaded payload is a valid, re-importable archive.
        root = unpack_archive(io.BytesIO(captured["payload"]), tmp_path / "out")
        assert len(list(read_archive(root))) == 3

    def test_push_target_dataset_override(self, monkeypatch):
        captured = {}

        class FakeClient:
            service_url = "https://fake.example"

            async def remember(self, data, dataset_name, **kwargs):
                captured["dataset_name"] = dataset_name
                return dict(_MIGRATION_RESPONSE)

            async def close(self):
                pass

        push_module = self._patch(
            monkeypatch, _fake_export(dataset_name="local_name"), FakeClient()
        )

        result = asyncio.run(push_module.push("local_name", target_dataset="remote_name"))
        assert captured["dataset_name"] == "remote_name"
        assert result.target_dataset == "remote_name"

    def test_push_empty_dataset_raises_before_upload(self, monkeypatch):
        class FakeClient:
            service_url = "https://fake.example"
            remember_called = False

            async def remember(self, data, dataset_name, **kwargs):
                FakeClient.remember_called = True
                return dict(_MIGRATION_RESPONSE)

            async def close(self):
                pass

        push_module = self._patch(monkeypatch, _fake_export(num_nodes=0, num_edges=0), FakeClient())

        with pytest.raises(ValueError, match="exported 0 nodes"):
            asyncio.run(push_module.push("main_dataset"))
        assert FakeClient.remember_called is False

    def test_push_rejects_non_migration_response(self, monkeypatch):
        """A pre-migration server ingests the tarball as a plain file; push must fail loudly."""

        class FakeClient:
            service_url = "https://fake.example"

            async def remember(self, data, dataset_name, **kwargs):
                return {"status": "completed", "items": [{"name": "main_dataset.cogx.tar.gz"}]}

            async def close(self):
                pass

        push_module = self._patch(monkeypatch, _fake_export(), FakeClient())

        with pytest.raises(RuntimeError, match="did not perform a COGX archive import"):
            asyncio.run(push_module.push("main_dataset"))

    def test_push_invalid_mode(self):
        from cognee.api.v1.push.push import push

        with pytest.raises(ValueError, match="Unknown push mode"):
            asyncio.run(push("main_dataset", mode="bogus"))

    def test_push_result_is_typed_dataclass(self, monkeypatch):
        """push() returns a PushResult dataclass, not a mutated raw dict."""
        import dataclasses

        from cognee.api.v1.push.push import PushResult

        class FakeClient:
            service_url = "https://fake.example"

            async def remember(self, data, dataset_name, **kwargs):
                return dict(_MIGRATION_RESPONSE)

            async def close(self):
                pass

        push_module = self._patch(monkeypatch, _fake_export(), FakeClient())
        result = asyncio.run(push_module.push("main_dataset"))

        assert isinstance(result, PushResult)
        assert dataclasses.is_dataclass(result)
        assert {field.name for field in dataclasses.fields(result)} == {
            "status",
            "dataset_name",
            "target_dataset",
            "num_nodes",
            "num_edges",
            "remote_response",
            "pipeline_run_id",
        }
        assert isinstance(result.status, str)
        assert isinstance(result.num_nodes, int)
        assert isinstance(result.num_edges, int)
        assert isinstance(result.remote_response, dict)


class TestCloudClientTimeout:
    """The push transport must survive uploads + imports longer than 5 minutes (H6)."""

    def test_session_default_keeps_total_timeout(self):
        """Ordinary API calls keep a bounded total; only uploads are unbounded."""
        import aiohttp

        from cognee.api.v1.serve.cloud_client import CloudClient

        async def get_timeout():
            client = CloudClient("https://fake.example", "key")
            session = await client._get_session()
            try:
                return session.timeout
            finally:
                await client.close()

        timeout = asyncio.run(get_timeout())
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == 300
        assert timeout.sock_connect == 30

    def test_upload_timeout_is_unbounded_and_scoped_to_archives(self):
        """The per-request archive-upload timeout drops the total deadline."""
        from cognee.api.v1.serve.cloud_client import CloudClient

        assert CloudClient.UPLOAD_TIMEOUT.total is None
        assert CloudClient.UPLOAD_TIMEOUT.sock_connect == 30
        assert CloudClient.UPLOAD_TIMEOUT.sock_read == 300
        assert CloudClient.DEFAULT_TIMEOUT.total == 300
