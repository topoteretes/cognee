import hashlib
import importlib
import io
import tarfile

import pytest

enola_module = importlib.import_module("cognee.tasks.code_graph.enola")
install_module = importlib.import_module("cognee.tasks.code_graph.install_enola")


def _make_archive(tmp_path, member_name, content=b"#!/bin/sh\nexit 0\n"):
    archive_path = tmp_path / f"{member_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return archive_path


def _pin_platform(monkeypatch, key="linux-amd64"):
    system, arch = key.split("-")
    machine = {"amd64": "x86_64", "arm64": "arm64"}[arch]
    monkeypatch.setattr(install_module.platform, "system", lambda: system)
    monkeypatch.setattr(install_module.platform, "machine", lambda: machine)


def _pin_download(monkeypatch, archive_path, checksum_key="linux-amd64", calls=None):
    monkeypatch.setitem(
        install_module.ENOLA_RELEASE_CHECKSUMS,
        checksum_key,
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )

    def fake_download(url, destination):
        if calls is not None:
            calls.append(url)
        destination.write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(install_module, "_download", fake_download)


def _expected_binary_name():
    return f"enola-{install_module.ENOLA_PINNED_VERSION}-linux-amd64"


def test_unsupported_platform_raises(monkeypatch):
    monkeypatch.setattr(install_module.platform, "system", lambda: "sunos")
    monkeypatch.setattr(install_module.platform, "machine", lambda: "sparc")

    with pytest.raises(install_module.EnolaInstallError) as exc_info:
        install_module.install_enola()

    assert "ENOLA_PATH" in str(exc_info.value)


def test_install_downloads_verifies_and_installs(monkeypatch, tmp_path):
    _pin_platform(monkeypatch)
    archive_path = _make_archive(tmp_path, _expected_binary_name())
    calls = []
    _pin_download(monkeypatch, archive_path, calls=calls)

    install_dir = tmp_path / "bin"
    binary_path = install_module.install_enola(install_dir=install_dir)

    assert binary_path == str(install_dir / _expected_binary_name())
    assert (install_dir / _expected_binary_name()).stat().st_mode & 0o111
    assert len(calls) == 1
    assert install_module.ENOLA_PINNED_VERSION in calls[0]


def test_install_is_idempotent_without_network(monkeypatch, tmp_path):
    _pin_platform(monkeypatch)
    archive_path = _make_archive(tmp_path, _expected_binary_name())
    calls = []
    _pin_download(monkeypatch, archive_path, calls=calls)
    install_dir = tmp_path / "bin"

    first = install_module.install_enola(install_dir=install_dir)
    second = install_module.install_enola(install_dir=install_dir)

    assert first == second
    assert len(calls) == 1


def test_checksum_mismatch_installs_nothing(monkeypatch, tmp_path):
    _pin_platform(monkeypatch)
    archive_path = _make_archive(tmp_path, _expected_binary_name())
    _pin_download(monkeypatch, archive_path)
    monkeypatch.setitem(install_module.ENOLA_RELEASE_CHECKSUMS, "linux-amd64", "0" * 64)
    install_dir = tmp_path / "bin"

    with pytest.raises(install_module.EnolaInstallError) as exc_info:
        install_module.install_enola(install_dir=install_dir)

    assert "Checksum mismatch" in str(exc_info.value)
    assert not (install_dir / _expected_binary_name()).exists()


def test_archive_with_unexpected_layout_is_refused(monkeypatch, tmp_path):
    _pin_platform(monkeypatch)
    archive_path = _make_archive(tmp_path, "../evil")
    _pin_download(monkeypatch, archive_path)
    install_dir = tmp_path / "bin"

    with pytest.raises(install_module.EnolaInstallError) as exc_info:
        install_module.install_enola(install_dir=install_dir)

    assert "refusing to extract" in str(exc_info.value)
    assert not (tmp_path / "evil").exists()


def test_http_disabled_refuses_download(monkeypatch, tmp_path):
    _pin_platform(monkeypatch)
    monkeypatch.setenv("ALLOW_HTTP_REQUESTS", "false")

    with pytest.raises(install_module.EnolaInstallError) as exc_info:
        install_module.install_enola(install_dir=tmp_path / "bin")

    assert "ALLOW_HTTP_REQUESTS" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_enola_generate_auto_installs_missing_binary(monkeypatch, tmp_path):
    monkeypatch.delenv("ENOLA_PATH", raising=False)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))

    fake_binary = tmp_path / "enola"
    fake_binary.write_text("#!/bin/sh\nexit 0\n")
    fake_binary.chmod(0o755)
    monkeypatch.setattr(install_module, "install_enola", lambda: str(fake_binary))

    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    commands = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        commands.append(args)
        snapshot_dir = repo_path / ".enola"
        snapshot_dir.mkdir(exist_ok=True)
        (snapshot_dir / "facts.jsonl").write_text('{"kind": "module", "name": "app"}\n')

        class _Process:
            returncode = 0

            async def communicate(self):
                return b"", b""

        return _Process()

    monkeypatch.setattr(enola_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    snapshot_dir = await enola_module.run_enola_generate(repo_path)

    assert snapshot_dir == repo_path / ".enola"
    assert commands[0][0] == str(fake_binary)


@pytest.mark.asyncio
async def test_run_enola_generate_respects_auto_install_opt_out(monkeypatch, tmp_path):
    monkeypatch.delenv("ENOLA_PATH", raising=False)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    monkeypatch.setenv("ENOLA_AUTO_INSTALL", "false")

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with pytest.raises(enola_module.EnolaNotInstalledError):
        await enola_module.run_enola_generate(repo_path)


@pytest.mark.asyncio
async def test_run_enola_generate_does_not_auto_install_over_bad_enola_path(monkeypatch, tmp_path):
    monkeypatch.setenv("ENOLA_PATH", str(tmp_path / "does_not_exist"))

    def unexpected_install(*args, **kwargs):
        raise AssertionError("auto-install must not run when ENOLA_PATH is set")

    monkeypatch.setattr(install_module, "install_enola", unexpected_install)

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    with pytest.raises(enola_module.EnolaNotInstalledError):
        await enola_module.run_enola_generate(repo_path)
