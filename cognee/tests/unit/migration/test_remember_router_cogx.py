"""Unit tests for the cogx-archive branch of the /v1/remember router.

All tests are pure: ``import_memory_source`` and dataset resolution are
monkeypatched, so no databases, LLM calls, or network are involved. Archive
unpacking and COGX parsing run for real against small in-memory tarballs.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.migration.archive import ARCHIVE_SUFFIX, pack_archive
from cognee.modules.migration.cogx import COGXArchiveWriter, COGXEntity, COGXFact
from cognee.modules.users.methods import get_authenticated_user

MOCK_USER = SimpleNamespace(id=uuid4(), email="test@example.com", is_active=True, tenant_id=uuid4())


def _packed_archive_bytes(tmp_path, name="sample"):
    """Build a real COGX archive (2 entities, 1 fact) and return tarball bytes."""
    archive_dir = tmp_path / f"{name}_cogx"
    with COGXArchiveWriter(archive_dir, source_system="cognee") as writer:
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
    tar_path = tmp_path / f"{name}{ARCHIVE_SUFFIX}"
    pack_archive(archive_dir, tar_path)
    return tar_path.read_bytes()


def _upload(payload, name="sample.cogx.tar.gz"):
    return ("data", (name, payload, "application/gzip"))


class FakeRememberResult:
    """Minimal stand-in for RememberResult, matching the .to_dict() contract."""

    def __init__(self, items_processed=3, items=None, dataset_name="test_dataset"):
        self.items_processed = items_processed
        self.items = (
            items
            if items is not None
            else [{"kind": "migration_import", "graph_nodes": 2, "graph_edges": 1}]
        )
        self.dataset_name = dataset_name

    def to_dict(self):
        result = {
            "status": "completed",
            "dataset_name": self.dataset_name,
            "items_processed": self.items_processed,
        }
        if self.items:
            result["items"] = self.items
        return result


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")

    async def override_user():
        return MOCK_USER

    app.dependency_overrides[get_authenticated_user] = override_user
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def fake_import(monkeypatch):
    """Patch import_memory_source on the migration package (lazily imported
    by the router at call time) with a capturing fake that consumes the
    source's record stream while the unpacked temp directory still exists."""
    import cognee.modules.migration as migration_pkg

    calls = []

    async def _fake_import_memory_source(source, dataset_name, user, run_in_background=False):
        records = [record async for record in source.records()]
        calls.append(
            {
                "source": source,
                "mode": source.mode,
                "dataset_name": dataset_name,
                "user": user,
                "run_in_background": run_in_background,
                "record_kinds": sorted(record.kind for record in records),
            }
        )
        return FakeRememberResult(dataset_name=dataset_name)

    _fake_import_memory_source.calls = calls
    monkeypatch.setattr(migration_pkg, "import_memory_source", _fake_import_memory_source)
    return _fake_import_memory_source


class TestCogxArchiveHappyPath:
    def test_single_archive_import(self, client, fake_import, tmp_path):
        payload = _packed_archive_bytes(tmp_path)

        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "content_type": "cogx-archive"},
            files=[_upload(payload)],
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["items_processed"] == 3
        assert any(item.get("kind") == "migration_import" for item in body["items"])

        assert len(fake_import.calls) == 1
        call = fake_import.calls[0]
        from cognee.modules.migration import COGXArchiveSource

        assert isinstance(call["source"], COGXArchiveSource)
        # No import_mode supplied: the router defaults to preserve.
        assert call["mode"] == "preserve"
        assert call["dataset_name"] == "test_dataset"
        assert call["user"] is MOCK_USER
        assert call["run_in_background"] is False
        # The archive was really unpacked and parsed (2 entities, 1 fact).
        assert call["record_kinds"] == ["entity", "entity", "fact"]

    def test_import_mode_forwarded_to_source(self, client, fake_import, tmp_path):
        payload = _packed_archive_bytes(tmp_path)

        resp = client.post(
            "/remember",
            data={
                "datasetName": "test_dataset",
                "content_type": "cogx-archive",
                "import_mode": "re-derive",
            },
            files=[_upload(payload)],
        )

        assert resp.status_code == 200
        assert fake_import.calls[0]["mode"] == "re-derive"

    def test_run_in_background_threaded_through(self, client, fake_import, tmp_path):
        payload = _packed_archive_bytes(tmp_path)

        resp = client.post(
            "/remember",
            data={
                "datasetName": "test_dataset",
                "content_type": "cogx-archive",
                "run_in_background": "true",
            },
            files=[_upload(payload)],
        )

        assert resp.status_code == 200
        assert fake_import.calls[0]["run_in_background"] is True

    def test_multi_archive_aggregate_result(self, client, monkeypatch, tmp_path):
        import cognee.modules.migration as migration_pkg

        results = [
            FakeRememberResult(items_processed=3, items=[{"kind": "migration_import", "n": 1}]),
            FakeRememberResult(items_processed=5, items=[{"kind": "migration_import", "n": 2}]),
        ]
        call_count = {"value": 0}

        async def fake_import_memory_source(source, dataset_name, user, run_in_background=False):
            result = results[call_count["value"]]
            call_count["value"] += 1
            return result

        monkeypatch.setattr(migration_pkg, "import_memory_source", fake_import_memory_source)

        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "content_type": "cogx-archive"},
            files=[
                _upload(_packed_archive_bytes(tmp_path, name="first"), name="first.cogx.tar.gz"),
                _upload(_packed_archive_bytes(tmp_path, name="second"), name="second.cogx.tar.gz"),
            ],
        )

        assert resp.status_code == 200
        assert call_count["value"] == 2
        body = resp.json()
        # items_processed is summed across archives, items are concatenated —
        # not just the last archive's counts.
        assert body["items_processed"] == 8
        assert [item["n"] for item in body["items"]] == [1, 2]


class TestCogxArchiveDatasetResolution:
    def test_dataset_id_only_resolves_name(self, client, fake_import, monkeypatch, tmp_path):
        import cognee.modules.pipelines.layers.resolve_authorized_user_datasets as resolve_module

        dataset_id = uuid4()
        resolver_calls = []

        async def fake_resolve(datasets, user):
            resolver_calls.append((datasets, user))
            return user, [SimpleNamespace(id=dataset_id, name="resolved_dataset")]

        monkeypatch.setattr(resolve_module, "resolve_authorized_user_datasets", fake_resolve)

        resp = client.post(
            "/remember",
            data={"datasetId": str(dataset_id), "content_type": "cogx-archive"},
            files=[_upload(_packed_archive_bytes(tmp_path))],
        )

        assert resp.status_code == 200
        assert resolver_calls == [(dataset_id, MOCK_USER)]
        assert fake_import.calls[0]["dataset_name"] == "resolved_dataset"


class TestCogxArchiveErrorMapping:
    def test_unknown_import_mode_returns_400(self, client, fake_import, tmp_path):
        resp = client.post(
            "/remember",
            data={
                "datasetName": "test_dataset",
                "content_type": "cogx-archive",
                "import_mode": "bogus",
            },
            files=[_upload(_packed_archive_bytes(tmp_path))],
        )

        assert resp.status_code == 400
        assert "bogus" in resp.json()["detail"]
        assert fake_import.calls == []

    def test_missing_file_returns_400(self, client, fake_import):
        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "content_type": "cogx-archive"},
        )

        assert resp.status_code == 400
        assert "archive" in resp.json()["detail"]
        assert fake_import.calls == []

    def test_garbage_bytes_returns_400(self, client, fake_import):
        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "content_type": "cogx-archive"},
            files=[_upload(b"this is not a tarball")],
        )

        # tarfile.TarError maps to 400, not the generic 409 handler.
        assert resp.status_code == 400
        assert "Invalid COGX archive" in resp.json()["error"]
        assert fake_import.calls == []

    def test_value_error_returns_400(self, client, monkeypatch, tmp_path):
        import cognee.modules.migration as migration_pkg

        async def raising_import(source, dataset_name, user, run_in_background=False):
            raise ValueError("archive written by a newer COGX version")

        monkeypatch.setattr(migration_pkg, "import_memory_source", raising_import)

        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "content_type": "cogx-archive"},
            files=[_upload(_packed_archive_bytes(tmp_path))],
        )

        # A validation failure maps to 400 with a generic body — the exception
        # text is logged server-side, never echoed to the caller (CodeQL
        # py/stack-trace-exposure).
        assert resp.status_code == 400
        assert resp.json()["error"] == "Invalid COGX archive."
        assert "newer COGX version" not in resp.json()["error"]

    def test_http_exception_not_swallowed_into_409(self, client, monkeypatch, tmp_path):
        import cognee.modules.migration as migration_pkg

        async def raising_import(source, dataset_name, user, run_in_background=False):
            raise HTTPException(status_code=403, detail="forbidden dataset")

        monkeypatch.setattr(migration_pkg, "import_memory_source", raising_import)

        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "content_type": "cogx-archive"},
            files=[_upload(_packed_archive_bytes(tmp_path))],
        )

        assert resp.status_code == 403
        assert resp.json()["detail"] == "forbidden dataset"

    def test_unexpected_error_returns_409(self, client, monkeypatch, tmp_path):
        import cognee.modules.migration as migration_pkg

        async def raising_import(source, dataset_name, user, run_in_background=False):
            raise RuntimeError("boom")

        monkeypatch.setattr(migration_pkg, "import_memory_source", raising_import)

        resp = client.post(
            "/remember",
            data={"datasetName": "test_dataset", "content_type": "cogx-archive"},
            files=[_upload(_packed_archive_bytes(tmp_path))],
        )

        assert resp.status_code == 409
        assert "COGX archive import" in resp.json()["error"]
