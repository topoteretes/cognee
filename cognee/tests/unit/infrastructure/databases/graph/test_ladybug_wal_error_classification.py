"""Only a real WAL replay failure may touch the WAL.

Open errors can name the .wal file without the WAL being corrupt: a lock held
by another process on a path that contains "wal", or permission denied on the
WAL itself. Treating those as corruption discarded a healthy WAL, and with it
every write since the last checkpoint.
"""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee_db_workers import kuzu_worker
from cognee_db_workers._kuzu_helpers import is_wal_replay_failure
from cognee_db_workers.harness import HandleRegistry


@pytest.mark.parametrize(
    "message",
    [
        "Runtime exception: Corrupted wal file. Read out invalid WAL record type.",
        "Runtime exception: Checksum verification failed, the WAL file is corrupted.",
        (
            "Runtime exception: WAL replay encountered an unknown record type (7). The WAL "
            "file is corrupted. Set throwOnWalReplayFailure=false to discard the tail and "
            "open the database."
        ),
        "Runtime exception: Invalid table type for insertion replay in WAL record.",
    ],
)
def test_replay_failures_are_recognized(message):
    assert is_wal_replay_failure(message, "/data/graph")


@pytest.mark.parametrize(
    "message",
    [
        "IO exception: Could not set lock on file : /Users/walter/db/graph",
        "IO exception: Cannot open file /srv/app/graph.wal: Permission denied",
        "IO exception: Cannot open file /srv/app/graph.wal: No space left on device",
        "Runtime exception: Database version mismatch",
    ],
)
def test_other_open_errors_are_not_replay_failures(message):
    assert not is_wal_replay_failure(message, "/srv/app/graph")


def test_db_path_never_matches():
    db_path = "/data/corrupted wal file/graph"
    assert not is_wal_replay_failure(
        f"IO exception: Could not set lock on file : {db_path}", db_path
    )


_HOLD_OPEN_WITH_UNCHECKPOINTED_WRITES = textwrap.dedent(
    """
    import sys
    import ladybug

    conn = ladybug.Connection(ladybug.Database(sys.argv[1]))
    conn.execute("CREATE NODE TABLE Item(id INT64, PRIMARY KEY(id))")
    conn.execute("CHECKPOINT")
    for i in range(50):
        conn.execute(f"CREATE (:Item {{id: {i}}})")
    print("ready", flush=True)
    sys.stdin.readline()
    """
)


@pytest.fixture
def locked_db_under_wal_path(tmp_path):
    # "walter" puts "wal" in every error message that names the path.
    directory = tmp_path / "walter"
    directory.mkdir()
    db_path = directory / "graph"
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_OPEN_WITH_UNCHECKPOINTED_WRITES, str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        wal_path = directory / "graph.wal"
        assert wal_path.exists() and wal_path.stat().st_size > 0
        yield str(db_path), wal_path, wal_path.read_bytes()
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)


def test_adapter_lock_error_keeps_live_wal(locked_db_under_wal_path):
    db_path, wal_path, wal_bytes = locked_db_under_wal_path

    with pytest.raises(RuntimeError, match="lock"):
        LadybugAdapter(db_path=db_path)

    assert wal_path.read_bytes() == wal_bytes
    assert not any(".corrupt" in name for name in os.listdir(wal_path.parent))


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
def db_with_unreadable_wal(tmp_path):
    db_path = tmp_path / "graph"
    subprocess.run([sys.executable, "-c", _WRITE_WITHOUT_CHECKPOINT, str(db_path)], check=True)
    wal_path = tmp_path / "graph.wal"
    wal_bytes = wal_path.read_bytes()
    wal_path.chmod(0)
    if os.access(wal_path, os.R_OK):
        wal_path.chmod(0o644)
        pytest.skip("running with privileges that ignore file permissions")
    try:
        yield str(db_path), wal_path, wal_bytes
    finally:
        if wal_path.exists():
            wal_path.chmod(0o644)


def _wal_untouched(wal_path, wal_bytes):
    wal_path.chmod(0o644)
    siblings = os.listdir(wal_path.parent)
    return wal_path.read_bytes() == wal_bytes and not any(".corrupt" in n for n in siblings)


def test_worker_permission_error_keeps_wal(db_with_unreadable_wal):
    db_path, wal_path, wal_bytes = db_with_unreadable_wal

    with pytest.raises(RuntimeError):
        kuzu_worker._open_database(
            HandleRegistry(), SimpleNamespace(kwargs={"database_path": db_path})
        )

    assert _wal_untouched(wal_path, wal_bytes)


def test_adapter_permission_error_keeps_wal(db_with_unreadable_wal):
    db_path, wal_path, wal_bytes = db_with_unreadable_wal

    with pytest.raises(RuntimeError):
        LadybugAdapter(db_path=db_path)

    assert _wal_untouched(wal_path, wal_bytes)


@pytest.fixture
def db_with_corrupt_wal(tmp_path):
    db_path = tmp_path / "graph"
    subprocess.run([sys.executable, "-c", _WRITE_WITHOUT_CHECKPOINT, str(db_path)], check=True)
    with open(tmp_path / "graph.wal", "ab") as wal:
        wal.write(b"\xff" * 64)
    return str(db_path)


def test_worker_still_recovers_from_corrupt_wal(db_with_corrupt_wal):
    registry = HandleRegistry()

    result = kuzu_worker._open_database(
        registry, SimpleNamespace(kwargs={"database_path": db_with_corrupt_wal})
    )

    assert registry.get(result.handle_id) is not None


def test_adapter_still_recovers_from_corrupt_wal(db_with_corrupt_wal):
    adapter = LadybugAdapter(db_path=db_with_corrupt_wal)

    assert adapter.connection is not None
