import subprocess
import sys
from types import SimpleNamespace

import pytest

from cognee.modules.pipelines import run_ownership as ownership


@pytest.fixture(autouse=True)
def system_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ownership, "get_base_config", lambda: SimpleNamespace(system_root_directory=str(tmp_path))
    )


def test_live_owner_cannot_be_claimed_and_terminal_markers_are_removed():
    with ownership.pipeline_run_ownership() as owner:
        run = SimpleNamespace(run_info={"recovery_lock": owner.token})
        with ownership.claim_run_ownership(run) as claim:
            assert claim is None
        owner.closed = True
    assert not ownership._lock_path(owner.token).exists()


def test_recovery_failure_keeps_marker_for_retry():
    with ownership.pipeline_run_ownership() as owner:
        run = SimpleNamespace(run_info={"recovery_lock": owner.token})
    for _ in range(2):
        with ownership.claim_run_ownership(run) as claim:
            assert claim is not None
    with ownership.claim_run_ownership(run) as claim:
        claim.closed = True
    assert not ownership._lock_path(owner.token).exists()


@pytest.mark.parametrize("token", [None, "../private-file", "00000000-0000-0000-0000-000000000000"])
def test_missing_or_invalid_ownership_is_not_claimed(token):
    with ownership.claim_run_ownership(SimpleNamespace(run_info={"recovery_lock": token})) as claim:
        assert claim is None


def test_os_releases_ownership_after_worker_death():
    with ownership.pipeline_run_ownership() as owner:
        path = ownership._lock_path(owner.token)
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "from filelock import FileLock; import sys; "
                "lock=FileLock(sys.argv[1]); lock.acquire(); print('owned', flush=True); "
                "sys.stdin.read()"
            ),
            str(path / "owner.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "owned"
        run = SimpleNamespace(run_info={"recovery_lock": owner.token})
        with ownership.claim_run_ownership(run) as claim:
            assert claim is None
        child.kill()
        child.wait(timeout=5)
        with ownership.claim_run_ownership(run) as claim:
            assert claim is not None
            claim.closed = True
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


def test_unwritable_system_directory_does_not_break_the_pipeline(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("read-only deployment")

    monkeypatch.setattr(ownership.Path, "mkdir", denied)
    with ownership.pipeline_run_ownership() as owner:
        assert owner.token is None
        owner.closed = True
