"""CSV ingestion routes through the loader engine's dlt_csv_loader.

With the dlt extra installed, the engine's priority order puts
``dlt_csv_loader`` above the plain ``csv_loader``, so every CSV takes the
structured DLT route: staging ingestion, one manifest per file with a stable
``data_id``, and the ``system_metadata`` route stamp — returned to
``ingest_data`` as a ``LoaderResult``. The text-flattening ``csv_loader``
remains reachable as an explicit per-call choice via ``preferred_loaders``,
and is the automatic fallback when dlt is not installed (this loader never
registers then).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

# The unit CI matrices install the base extras only — without dlt the loader
# under test is deliberately not registered.
pytest.importorskip("dlt")

import importlib  # noqa: E402

# The ingestion package re-exports these names, shadowing the submodules on
# plain ``import a.b.c as m`` — import_module returns the real modules.
ingest_dlt_source_module = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")
resolve_module = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")
from cognee.infrastructure.loaders.create_loader_engine import create_loader_engine  # noqa: E402
from cognee.infrastructure.loaders.external.dlt_csv_loader import DltCsvLoader  # noqa: E402
from cognee.infrastructure.loaders.LoaderInterface import LoaderResult  # noqa: E402
from cognee.modules.ingestion.exceptions import IngestionError  # noqa: E402
from cognee.tasks.ingestion.data_item import DataItem  # noqa: E402

CSV_BYTES = b"id,name\n1,Ada\n2,Alan\n"


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "people.csv"
    path.write_bytes(CSV_BYTES)
    return str(path)


class TestDispatch:
    def test_dlt_loader_wins_csv_dispatch(self, csv_file):
        engine = create_loader_engine()
        loader = engine.get_loader(csv_file, None)
        assert loader is not None
        assert loader.loader_name == "dlt_csv_loader"

    def test_preferred_loader_forces_plain_csv(self, csv_file):
        engine = create_loader_engine()
        loader = engine.get_loader(csv_file, {"csv_loader": {}})
        assert loader is not None
        assert loader.loader_name == "csv_loader"


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_returns_manifest_loader_result(self, csv_file, tmp_path, monkeypatch):
        """load() must run staging ingestion + manifest build and hand back
        the manifest's stable id and route stamp as a LoaderResult."""
        manifest_id = uuid4()
        stamp = {"source": "dlt_source", "source_name": "people", "row_count": 2}
        manifest_item = DataItem(data='{"rows": []}', data_id=manifest_id, system_metadata=stamp)

        captured = {}

        def fake_create(path, source_name=None):
            captured["source_name"] = source_name
            captured["path"] = path
            return SimpleNamespace(name=source_name)

        import cognee.tasks.ingestion.create_dlt_source as create_module
        import cognee.infrastructure.loaders.external.dlt_csv_loader as loader_module

        monkeypatch.setattr(create_module, "create_dlt_source_from_csv", fake_create)
        monkeypatch.setattr(
            ingest_dlt_source_module, "ingest_dlt_source", AsyncMock(return_value=["row"])
        )
        monkeypatch.setattr(
            resolve_module, "_build_source_manifest_item", AsyncMock(return_value=manifest_item)
        )

        stored = {}

        def fake_get_file_storage(root):
            async def store(name, content):
                stored["name"] = name
                stored["content"] = content
                return str(tmp_path / name)

            return SimpleNamespace(store=store)

        monkeypatch.setattr(loader_module, "get_file_storage", fake_get_file_storage)
        monkeypatch.setattr(
            loader_module,
            "get_storage_config",
            lambda: {"data_root_directory": str(tmp_path)},
        )

        result = await DltCsvLoader().load(
            csv_file,
            dataset_name="ds",
            user=SimpleNamespace(id=uuid4()),
            original_file_name="Lab Machines.csv",
        )

        assert isinstance(result, LoaderResult)
        assert result.data_id == manifest_id
        assert result.system_metadata == stamp
        assert stored["content"] == '{"rows": []}'
        assert result.file_path.endswith(stored["name"])
        # The manifest identity derives from the ORIGINAL file name, not the
        # (possibly temp/stored) copy the loader actually read.
        assert captured["source_name"] == "lab_machines"
        assert captured["path"] == csv_file

    @pytest.mark.asyncio
    async def test_load_without_ingestion_context_fails_loudly(self, csv_file):
        with pytest.raises(IngestionError):
            await DltCsvLoader().load(csv_file)

    @pytest.mark.asyncio
    async def test_empty_source_fails_loudly(self, csv_file, monkeypatch):
        monkeypatch.setattr(
            ingest_dlt_source_module, "ingest_dlt_source", AsyncMock(return_value=[])
        )
        monkeypatch.setattr(
            resolve_module, "_build_source_manifest_item", AsyncMock(return_value=None)
        )
        with pytest.raises(IngestionError, match="no rows"):
            await DltCsvLoader().load(csv_file, dataset_name="ds", user=SimpleNamespace(id=uuid4()))
