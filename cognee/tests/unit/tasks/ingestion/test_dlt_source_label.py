"""A connection-string source is labelled with the database it reads, never with
the host or credentials of its connection string."""

import pytest

from cognee.tasks.ingestion.create_dlt_source import _database_label


@pytest.mark.parametrize(
    ("connection_string", "label"),
    [
        ("sqlite:////data/shop/hr.sqlite", "hr.sqlite"),
        ("sqlite:///relative/crm.db", "crm.db"),
        ("postgresql://admin:s3cret@db.internal:5432/crm", "crm"),
        ("mysql+pymysql://user:pw@localhost/inventory", "inventory"),
    ],
)
def test_the_label_is_the_database_name(connection_string, label):
    assert _database_label(connection_string) == label


def test_the_label_never_carries_credentials():
    assert "s3cret" not in _database_label("postgresql://admin:s3cret@db.internal:5432/crm")
