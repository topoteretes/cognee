"""Hash-partitioned pgvector collections: names and the setting (no database)."""

import pytest

# The adapter imports asyncpg and pgvector, which only the postgres extra installs.
pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from cognee.infrastructure.databases.vector.config import VectorConfig
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    _VECTOR_COLLECTIONS_SQL,
    partition_name,
)


def test_a_partition_is_named_after_its_collection_and_remainder():
    assert partition_name("Entity_name", 0) == "Entity_name_p0"
    assert partition_name("Entity_name", 15) == "Entity_name_p15"


def test_partitions_of_long_collection_names_stay_distinct_and_within_the_limit():
    """Postgres truncates an identifier at 63 bytes: two collections sharing their first
    bytes would otherwise get the same partition names, and the second ``CREATE TABLE``
    would fail."""
    first, second = "C" * 70 + "_first", "C" * 70 + "_second"

    names = [partition_name(name, remainder) for name in (first, second) for remainder in (0, 7)]

    assert len(set(names)) == 4
    assert all(len(name.encode()) <= 63 for name in names)
    assert [name.rsplit("_", 1)[-1] for name in names] == ["p0", "p7", "p0", "p7"]


def test_a_partitioned_collection_is_listed_as_its_parent_table_only():
    assert "c.relkind IN ('r', 'p')" in _VECTOR_COLLECTIONS_SQL
    assert "NOT c.relispartition" in _VECTOR_COLLECTIONS_SQL


def test_collections_are_one_table_unless_the_setting_says_otherwise(monkeypatch):
    monkeypatch.delenv("PGVECTOR_COLLECTION_PARTITIONS", raising=False)
    assert VectorConfig().pgvector_collection_partitions == 1

    monkeypatch.setenv("PGVECTOR_COLLECTION_PARTITIONS", "16")
    assert VectorConfig().pgvector_collection_partitions == 16


@pytest.mark.parametrize("value", ["0", "-4", "many"])
def test_the_number_of_partitions_is_an_integer_of_at_least_one(monkeypatch, value):
    monkeypatch.setenv("PGVECTOR_COLLECTION_PARTITIONS", value)
    with pytest.raises(ValueError, match="pgvector_collection_partitions"):
        VectorConfig()
