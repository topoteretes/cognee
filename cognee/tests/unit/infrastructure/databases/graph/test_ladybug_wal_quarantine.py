"""A WAL that fails to replay must be moved aside, not deleted.

The WAL holds committed transactions that are not yet checkpointed. When
replay fails, both open paths reopen the database without it; the original
bytes must stay on disk so an operator can recover those transactions.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee_db_workers import kuzu_worker
from cognee_db_workers.harness import HandleRegistry

_WRITE_WITHOUT_CHECKPOINT = textwrap.dedent(
    """
    import os, sys
    import ladybug

    conn = ladybug.Connection(ladybug.Database(sys.argv[1]))
    conn.execute("CREATE NODE TABLE Item(id INT64, PRIMARY KEY(id))")
    conn.execute("CHECKPOINT")
    for i in range(50):
        conn.execute(f"CREATE (:Item {{id: {i}}})")
    os._exit(0)
    """
)


@pytest.fixture
def db_with_unreplayable_wal(tmp_path):
    db_path = tmp_path / "graph"
    subprocess.run([sys.executable, "-c", _WRITE_WITHOUT_CHECKPOINT, str(db_path)], check=True)
    wal_path = tmp_path / "graph.wal"
    with open(wal_path, "ab") as wal:
        wal.write(b"\xff" * 64)
    return str(db_path), wal_path.read_bytes()


def _files_with_bytes(directory, content):
    return [name for name in os.listdir(directory) if (directory / name).read_bytes() == content]


def test_worker_open_keeps_unreplayable_wal(db_with_unreplayable_wal, tmp_path):
    db_path, wal_bytes = db_with_unreplayable_wal
    registry = HandleRegistry()

    result = kuzu_worker._open_database(
        registry, SimpleNamespace(kwargs={"database_path": db_path})
    )

    assert registry.get(result.handle_id) is not None
    assert len(_files_with_bytes(tmp_path, wal_bytes)) == 1, "WAL bytes were discarded"


def test_adapter_open_keeps_unreplayable_wal(db_with_unreplayable_wal, tmp_path):
    db_path, wal_bytes = db_with_unreplayable_wal

    adapter = LadybugAdapter(db_path=db_path)

    assert adapter.connection is not None
    assert len(_files_with_bytes(tmp_path, wal_bytes)) == 1, "WAL bytes were discarded"
