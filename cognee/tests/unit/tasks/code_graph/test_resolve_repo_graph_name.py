"""Two repositories that share a readable name must not share a code graph."""

import asyncio
import importlib
import os
import stat

import pytest

resolve_module = importlib.import_module("cognee.tasks.code_graph.resolve_repo")
CodeRepositoryError = resolve_module.CodeRepositoryError

_FIRST = "https://github.com/a-b/c"
_SECOND = "https://github.com/a/b-c"


@pytest.mark.asyncio
async def test_a_repository_sharing_a_readable_name_is_refused(tmp_path):
    await resolve_module._claim_graph_name(tmp_path, _FIRST)
    await resolve_module._claim_graph_name(tmp_path, _FIRST)  # the owner may come back

    with pytest.raises(CodeRepositoryError, match="already used by"):
        await resolve_module._claim_graph_name(tmp_path, _SECOND)


@pytest.mark.asyncio
async def test_a_repository_stored_before_claims_keeps_its_name(tmp_path, monkeypatch):
    """The old clone directory (the readable slug) says who already owns the name."""
    legacy = tmp_path / resolve_module._readable_slug(_SECOND)
    (legacy / ".git").mkdir(parents=True)

    async def origin(clone):
        return _SECOND

    monkeypatch.setattr(resolve_module, "_origin_url", origin)

    with pytest.raises(CodeRepositoryError, match="already used by"):
        await resolve_module._claim_graph_name(tmp_path, _FIRST)
    await resolve_module._claim_graph_name(tmp_path, _SECOND)


@pytest.mark.skipif(os.name != "posix", reason="Checks POSIX directory modes")
def test_a_partial_clone_has_the_mode_of_the_clones_directory(tmp_path):
    tmp_path.chmod(0o755)
    partial = resolve_module._make_partial_clone_dir(tmp_path / "clone")
    assert stat.S_IMODE(partial.stat().st_mode) == 0o755


@pytest.mark.asyncio
async def test_case_variants_get_their_own_claims(tmp_path):
    """'Org' and 'org' are different graph names, even on a case-insensitive file system."""
    await resolve_module._claim_graph_name(tmp_path, "https://github.com/o/r0")
    await resolve_module._claim_graph_name(tmp_path, "https://github.com/O/R0")


@pytest.mark.asyncio
async def test_a_claim_leaks_no_file_descriptor(tmp_path):
    descriptors = "/dev/fd" if os.path.isdir("/dev/fd") else "/proc/self/fd"
    before = len(os.listdir(descriptors))
    for index in range(20):
        await resolve_module._claim_graph_name(tmp_path, f"https://github.com/o/r{index}")
    assert len(os.listdir(descriptors)) == before


@pytest.mark.asyncio
async def test_a_file_system_without_hard_links_raises_a_repository_error(tmp_path, monkeypatch):
    def no_link(source, destination):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(os, "link", no_link)
    with pytest.raises(CodeRepositoryError, match="Could not record"):
        await resolve_module._claim_graph_name(tmp_path, _FIRST)
    assert not any((tmp_path / ".names").iterdir())


@pytest.mark.asyncio
async def test_a_cancelled_cleanup_that_fails_still_cancels():
    async def failing_cleanup():
        await asyncio.sleep(0.05)
        raise RuntimeError("cleanup failed")

    task = asyncio.ensure_future(resolve_module._finish_cleanup(failing_cleanup()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
