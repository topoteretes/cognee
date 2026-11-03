"""Under backend access control the pgvector engine with no database name is the
deployment's default store: the relational database.

The dataset handlers name every dataset's database, so an unnamed engine can only
be the default store (dataset routing keeps its collection there). Connecting with
an empty name made asyncpg pick a database named after the role, which does not
exist.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cognee.infrastructure.databases.vector.create_vector_engine as cve

pytest.importorskip("asyncpg")
pgvector_module = importlib.import_module(
    "cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter"
)
context_module = importlib.import_module("cognee.context_global_variables")
relational_module = importlib.import_module("cognee.infrastructure.databases.relational")


def _create(name: str, host: str) -> str:
    """The connection string the pgvector adapter is built with."""
    captured = {}

    class _CapturingAdapter:
        def __init__(self, connection_string, api_key, embedding_engine, schema=""):
            captured["connection_string"] = connection_string

    with (
        patch.object(cve, "get_embedding_engine", return_value=MagicMock()),
        patch.object(pgvector_module, "PGVectorAdapter", _CapturingAdapter),
        patch.object(context_module, "backend_access_control_enabled", lambda: True),
        patch.object(
            relational_module,
            "get_relational_config",
            lambda: SimpleNamespace(db_name="cognee_db"),
        ),
    ):
        cve._create_vector_engine(
            vector_db_provider="pgvector",
            vector_db_url="",
            vector_db_name=name,
            vector_db_port="5432",
            vector_db_key="",
            vector_dataset_database_handler="pgvector",
            vector_db_username="cognee",
            vector_db_password="secret",
            vector_db_host=host,
            vector_db_subprocess_enabled=True,
        )
    return captured["connection_string"]


def test_an_unnamed_engine_is_the_relational_database():
    assert _create("", "default-store-host").endswith("@default-store-host:5432/cognee_db")


def test_a_named_engine_keeps_its_name():
    assert _create("dataset-db", "dataset-store-host").endswith(
        "@dataset-store-host:5432/dataset-db"
    )
