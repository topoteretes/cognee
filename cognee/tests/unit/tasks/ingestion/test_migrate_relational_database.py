import importlib
from types import SimpleNamespace

import pytest

from cognee.modules.engine.models import ColumnValue

migration_module = importlib.import_module("cognee.tasks.ingestion.migrate_relational_database")


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Cursor:
    async def execute(self, query):
        query = str(query)
        if query == "SELECT * FROM companies;":
            return _Result([(7, "Acme")])
        if query == "SELECT * FROM employees;":
            return _Result([(10, 42, "Ann", 7)])
        if "JOIN companies AS companies_e2" in query:
            return _Result([(10, 7)])
        raise AssertionError(f"Unexpected migration query: {query}")


class _EngineContext:
    async def __aenter__(self):
        return _Cursor()

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


@pytest.mark.asyncio
async def test_column_migration_skips_local_foreign_key_and_primary_key_columns(monkeypatch):
    engine = SimpleNamespace(engine=SimpleNamespace(begin=lambda: _EngineContext()))
    monkeypatch.setattr(migration_module, "get_migration_relational_engine", lambda: engine)

    # Keep equal string values in separate objects to cover value equality over identity.
    primary_key_parts = ["employee", "_id"]
    primary_key = "".join(primary_key_parts)
    column_primary_key = "".join(primary_key_parts)
    assert primary_key == column_primary_key
    assert primary_key is not column_primary_key

    schema = {
        "companies": {
            "columns": [{"name": "id"}, {"name": "name"}],
            "primary_key": "id",
            "foreign_keys": [],
        },
        "employees": {
            "columns": [
                {"name": column_primary_key},
                {"name": "id"},
                {"name": "name"},
                {"name": "company_id"},
            ],
            "primary_key": primary_key,
            "foreign_keys": [
                {"column": "company_id", "ref_table": "companies", "ref_column": "id"}
            ],
        },
    }

    node_mapping, edge_mapping = await migration_module.complete_database_ingestion(
        schema, migrate_column_data=True
    )

    column_value_names = {
        node.name for node in node_mapping.values() if isinstance(node, ColumnValue)
    }
    assert column_value_names == {
        "employees:id:42",
        "employees:name:Ann",
        "companies:name:Acme",
    }

    employee_id = node_mapping["employees:10"].id
    company_id = node_mapping["companies:7"].id
    assert any(
        source_id == employee_id and target_id == company_id and relationship_name == "company_id"
        for source_id, target_id, relationship_name, _ in edge_mapping
    )
