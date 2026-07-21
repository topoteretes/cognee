import asyncio
import importlib

import pytest

enola_module = importlib.import_module("cognee.tasks.code_graph.enola")


def _make_fake_binary(tmp_path):
    fake_binary = tmp_path / "enola"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)
    return fake_binary


def test_find_enola_binary_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("ENOLA_PATH", raising=False)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    with pytest.raises(enola_module.EnolaNotInstalledError):
        enola_module.find_enola_binary()


def test_find_enola_binary_respects_enola_path_override(monkeypatch, tmp_path):
    fake_binary = _make_fake_binary(tmp_path)
    monkeypatch.setenv("ENOLA_PATH", str(fake_binary))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    assert enola_module.find_enola_binary() == str(fake_binary)


def test_find_enola_binary_invalid_enola_path_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("ENOLA_PATH", str(tmp_path / "does_not_exist"))

    with pytest.raises(enola_module.EnolaNotInstalledError):
        enola_module.find_enola_binary()


class _FakeProcess:
    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self):
        return b"", self._stderr

    def kill(self):
        pass

    async def wait(self):
        pass


@pytest.mark.asyncio
async def test_run_enola_generate_nonzero_exit_raises_with_stderr(monkeypatch, tmp_path):
    fake_binary = _make_fake_binary(tmp_path)
    monkeypatch.setenv("ENOLA_PATH", str(fake_binary))
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(returncode=3, stderr=b"parse failure: boom")

    monkeypatch.setattr(enola_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(enola_module.EnolaSnapshotError) as exc_info:
        await enola_module.run_enola_generate(repo_path)

    assert "boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_enola_generate_missing_facts_raises(monkeypatch, tmp_path):
    fake_binary = _make_fake_binary(tmp_path)
    monkeypatch.setenv("ENOLA_PATH", str(fake_binary))
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _FakeProcess(returncode=0)

    monkeypatch.setattr(enola_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(enola_module.EnolaSnapshotError):
        await enola_module.run_enola_generate(repo_path)


@pytest.mark.asyncio
async def test_run_enola_generate_returns_snapshot_dir(monkeypatch, tmp_path):
    fake_binary = _make_fake_binary(tmp_path)
    monkeypatch.setenv("ENOLA_PATH", str(fake_binary))
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    async def fake_create_subprocess_exec(*args, **kwargs):
        snapshot_dir = repo_path / ".enola"
        snapshot_dir.mkdir(exist_ok=True)
        (snapshot_dir / "facts.jsonl").write_text('{"kind": "module", "name": "app"}\n')
        return _FakeProcess(returncode=0)

    monkeypatch.setattr(enola_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    snapshot_dir = await enola_module.run_enola_generate(repo_path)

    assert snapshot_dir == repo_path / ".enola"
    assert (snapshot_dir / "facts.jsonl").is_file()
