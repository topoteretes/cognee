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
    # The enola-cli wheel may be installed in the test environment; hide it too.
    monkeypatch.setattr(enola_module.sysconfig, "get_path", lambda name: str(empty_dir))

    with pytest.raises(enola_module.EnolaNotInstalledError):
        enola_module.find_enola_binary()


def test_find_enola_binary_respects_enola_path_override(monkeypatch, tmp_path):
    fake_binary = _make_fake_binary(tmp_path)
    monkeypatch.setenv("ENOLA_PATH", str(fake_binary))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    assert enola_module.find_enola_binary() == str(fake_binary)


@pytest.mark.parametrize(
    ("system", "binary_name"),
    [("Linux", "enola"), ("Darwin", "enola"), ("Windows", "enola.exe")],
)
def test_find_enola_binary_prefers_environment_scripts_dir(
    monkeypatch, tmp_path, system, binary_name
):
    """The enola-cli wheel installs the binary next to the interpreter; that wins over PATH.

    The wheel's console-script shim is ``enola.exe`` on Windows and ``enola``
    everywhere else, and the lookup builds that name from ``platform.system()``.
    The platform is driven here rather than inherited from the host so all three
    names are covered on every runner: a fixture that only matched the host's
    naming passed on Linux and macOS while failing on Windows for the whole
    lifetime of the test.
    """
    monkeypatch.delenv("ENOLA_PATH", raising=False)
    monkeypatch.setattr(enola_module.platform, "system", lambda: system)
    scripts_dir = tmp_path / "venv-bin"
    scripts_dir.mkdir()
    wheel_binary = scripts_dir / binary_name
    wheel_binary.write_text("#!/bin/sh\n")
    wheel_binary.chmod(0o755)
    monkeypatch.setattr(enola_module.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(enola_module.shutil, "which", lambda _name: "/usr/local/bin/enola")

    assert enola_module.find_enola_binary() == str(wheel_binary)


def test_find_enola_binary_falls_back_to_path_when_scripts_dir_has_none(monkeypatch, tmp_path):
    monkeypatch.delenv("ENOLA_PATH", raising=False)
    monkeypatch.setattr(enola_module.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(enola_module.shutil, "which", lambda _name: "/usr/local/bin/enola")

    assert enola_module.find_enola_binary() == "/usr/local/bin/enola"


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
