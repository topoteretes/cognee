"""The GLiNER runtime install: torch (CPU build) only when missing, then the gliner extra."""

import asyncio
import logging
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cognee.modules.cognify import config as cognify_config_module
from cognee.tasks.graph.gliner_demo import install

COGNEE_REQUIRES = [
    "filelock>=3.12.0,<4.0.0",
    'numpy>=1.26; python_version >= "3.10"',
    'gliner2[local]<3,>=2.0.0; extra == "gliner"',
    'torch<3,>=2.1; extra == "gliner"',
    'protobuf>=5.29.6; extra == "gliner"',
    'boto3>=1.26; extra == "aws"',
    'torch>=2.5; extra == "evals"',
]


def test_the_gliner_extra_comes_from_cognee_metadata_with_torch_split_out():
    with patch("importlib.metadata.requires", return_value=COGNEE_REQUIRES):
        assert install.gliner_extra() == (
            "torch<3,>=2.1",
            ["gliner2[local]<3,>=2.0.0", "protobuf>=5.29.6"],
        )


def test_the_real_gliner_extra_declares_torch():
    torch, rest = install.gliner_extra()
    assert torch.startswith("torch")
    assert any(requirement.startswith("gliner2") for requirement in rest)


def test_an_incomplete_extra_raises_with_the_extra_as_the_fix():
    with (
        patch("importlib.metadata.requires", return_value=COGNEE_REQUIRES[:3]),
        pytest.raises(install.GlinerInstallError, match="no complete `gliner` extra") as raised,
    ):
        install.gliner_extra()
    assert raised.value.remediation.startswith(
        'Install cognee with the GLiNER extra: pip install "cognee[gliner]"'
    )


def test_installer_prefers_pip_then_uv_then_raises():
    with patch("importlib.util.find_spec", return_value=object()):
        assert install.installer_command() == ("pip", [sys.executable, "-m", "pip", "install"])
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("shutil.which", return_value="/bin/uv"),
    ):
        assert install.installer_command() == (
            "uv",
            ["/bin/uv", "pip", "install", "--python", sys.executable],
        )
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("shutil.which", return_value=None),
        pytest.raises(
            install.GlinerInstallError, match=r"neither pip nor uv(.|\n)*cognee\[gliner\]"
        ),
    ):
        install.installer_command()


def _install(tmp_path, present, installed_after=True):
    """Run install_gliner_runtime with the environment faked.

    ``present`` is the set of modules importable before the install; returns the
    steps run as ``(command, step, constraints)``, the constraints file read while
    it existed.
    """
    runs = []

    def run(command, step, failed_step, installer):
        constraints = None
        if "-c" in command:
            constraints = Path(command[command.index("-c") + 1]).read_text().split()
        runs.append((command, step, constraints))

    def runtime_installed():
        return {"gliner2", "torch"} <= present

    def verify(installer):
        if not installed_after:
            raise install.GlinerInstallError("cannot import it (boom).", "verify", installer)

    with (
        patch.object(sys, "prefix", str(tmp_path)),
        patch.object(install, "_installed", side_effect=lambda module: module in present),
        patch.object(install, "gliner_runtime_installed", side_effect=runtime_installed),
        patch.object(install, "installer_command", return_value=("pip", ["pip", "install"])),
        patch.object(install, "_run", side_effect=run),
        patch.object(install, "_import_runtime", side_effect=verify),
        patch.object(install, "installed_pins", return_value=["numpy==2.2.0"]),
        patch.object(
            install, "gliner_extra", return_value=("torch<3,>=2.1", ["gliner2[local]<3,>=2.0.0"])
        ),
        patch("importlib.metadata.version", return_value="2.9.1+cpu"),
    ):
        outcome = install.install_gliner_runtime("https://mirror.example/cpu")
    return runs, outcome


def test_a_shadowed_distribution_pins_the_copy_python_imports():
    """A venv with --system-site-packages lists the venv's copy first, then the
    system's; import resolves the first, so the pin must not be the second."""

    def dist(name, version):
        return SimpleNamespace(metadata={"Name": name}, version=version)

    with patch(
        "importlib.metadata.distributions",
        return_value=[dist("numpy", "2.2.0"), dist("Numpy", "1.26.4"), dist("tqdm", "4.67.1")],
    ):
        assert install.installed_pins() == ["numpy==2.2.0", "tqdm==4.67.1"]


def test_missing_torch_comes_from_the_cpu_index_then_the_extra_with_installed_pins(tmp_path):
    (torch_step, gliner_step), outcome = _install(tmp_path, present=set())
    assert (outcome.installer, outcome.installed, outcome.torch_version) == (
        "pip",
        ["torch", "gliner2"],
        "2.9.1+cpu",
    )
    assert torch_step[0] == [
        "pip",
        "install",
        "--index-url",
        "https://mirror.example/cpu",
        "torch<3,>=2.1",
    ]
    assert "step 1 of 2" in torch_step[1]
    command, step, constraints = gliner_step
    assert command[:3] == ["pip", "install", "-c"]
    assert command[-1] == "gliner2[local]<3,>=2.0.0"
    assert constraints == ["numpy==2.2.0"]
    assert "step 2 of 2" in step


def test_an_installed_torch_cpu_or_gpu_is_kept(tmp_path):
    runs, outcome = _install(tmp_path, present={"torch"})
    assert outcome.installed == ["gliner2"]
    assert [step for _, step, _ in runs] == [
        "installing gliner2 and its tokenizer dependencies (step 2 of 2)"
    ]


def test_install_is_a_no_op_when_the_runtime_is_present(tmp_path):
    runs, outcome = _install(tmp_path, present={"torch", "gliner2"})
    assert runs == [] and outcome.installed == [] and outcome.installer is None


def test_install_raises_when_the_runtime_is_still_not_importable(tmp_path):
    with pytest.raises(install.GlinerInstallError, match="cannot import it") as raised:
        _install(tmp_path, present=set(), installed_after=False)
    assert (raised.value.step, raised.value.installer) == ("verify", "pip")


def test_a_successful_install_suggests_the_extra(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _install(tmp_path, present=set())
    logged = " | ".join(record.getMessage() for record in caplog.records)
    assert (
        'Tip: if you keep using GLiNER, install cognee with the extra (pip install "cognee[gliner]")'
        in logged
    )


def test_an_unwritable_environment_raises_with_the_extra_as_the_fix(tmp_path):
    # The lock fails the way it does in a read-only prefix. Simulated rather than
    # made with chmod, which Windows does not enforce on directories.
    denied = PermissionError(13, "Permission denied", str(tmp_path / ".cognee-gliner-install.lock"))
    with (
        patch.object(sys, "prefix", str(tmp_path)),
        patch.object(install.FileLock, "acquire", side_effect=denied),
        patch.object(install, "_run", side_effect=AssertionError("must not install")),
        pytest.raises(install.GlinerInstallError, match="is not writable") as raised,
    ):
        install.install_gliner_runtime("https://mirror.example/cpu")
    assert raised.value.step == "lock"
    assert "cognee[gliner]" in raised.value.remediation


def test_the_constraints_file_is_removed_even_when_writing_it_fails(tmp_path):
    with (
        patch.object(tempfile, "tempdir", str(tmp_path)),
        patch.object(install, "installed_pins", side_effect=RuntimeError("bad metadata")),
        pytest.raises(RuntimeError, match="bad metadata"),
    ):
        install._install_rest(["pip", "install"], ["gliner2"], "pip")
    assert list(tmp_path.iterdir()) == []


def test_run_relays_progress_lines_and_raises_with_the_output_tail(caplog):
    script = (
        "print('Collecting torch'); print('Requirement already satisfied: x');"
        "print('Successfully installed torch-2.9.1'); raise SystemExit(0)"
    )
    with caplog.at_level(logging.INFO):
        install._run([sys.executable, "-c", script], "downloading CPU-only PyTorch", "torch", "pip")
    logged = " | ".join(record.getMessage() for record in caplog.records)
    assert "GLiNER runtime: Collecting torch" in logged
    assert "GLiNER runtime: Successfully installed torch-2.9.1" in logged
    assert "Requirement already satisfied" not in logged
    assert "GLiNER runtime: done downloading CPU-only PyTorch in" in logged

    failing = "print('ERROR: No matching distribution found for torch'); raise SystemExit(1)"
    with pytest.raises(install.GlinerInstallError, match="No matching distribution") as raised:
        install._run([sys.executable, "-c", failing], "downloading CPU-only PyTorch", "torch", "uv")
    assert (raised.value.step, raised.value.installer) == ("torch", "uv")


def test_run_reports_elapsed_time_while_a_step_is_slow(caplog):
    with patch.object(install, "HEARTBEAT_SECONDS", 0.05), caplog.at_level(logging.INFO):
        install._run(
            [sys.executable, "-c", "import time; time.sleep(0.3)"], "downloading", "torch", "pip"
        )
    assert any("still downloading" in record.getMessage() for record in caplog.records)


# --- ensure_extractor_runtime: the awaited, off-loop install and its telemetry ---


def _config(**overrides):
    return cognify_config_module.CognifyConfig().model_copy(update=overrides)


@pytest.fixture
def events(monkeypatch):
    sent = []
    monkeypatch.setattr(
        "cognee.shared.utils.send_telemetry",
        lambda name, user=None, additional_properties=None: sent.append(
            (name, additional_properties)
        ),
    )
    return sent


@pytest.mark.asyncio
async def test_ensure_is_a_no_op_for_the_llm_extractor_and_a_present_runtime(events):
    with patch.object(install, "install_gliner_runtime") as run:
        await cognify_config_module.ensure_extractor_runtime("llm", _config())
        with patch.object(install, "gliner_runtime_installed", return_value=True):
            await cognify_config_module.ensure_extractor_runtime("gliner_demo", _config())
    run.assert_not_called()
    assert events == []


@pytest.mark.asyncio
async def test_ensure_raises_the_hint_when_auto_install_is_off(events):
    with (
        patch.object(install, "gliner_runtime_installed", return_value=False),
        patch.object(install, "install_gliner_runtime") as run,
        pytest.raises(cognify_config_module.KeylessExtractorNotInstalledError),
    ):
        await cognify_config_module.ensure_extractor_runtime(
            "gliner_demo", _config(gliner_auto_install=False)
        )
    run.assert_not_called()
    assert events == []


@pytest.mark.asyncio
async def test_the_install_runs_off_the_event_loop_and_the_caller_waits_for_it(events):
    """Other coroutines keep running during the install; the caller resumes only after it."""
    installing = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    install_thread = []

    def slow_install(index_url, on_start):
        install_thread.append(threading.get_ident())
        on_start()
        installing.set()
        assert release.wait(5)
        return install.InstallOutcome("pip", ["torch", "gliner2"], "2.9.1+cpu", 42)

    ticks = 0

    async def other_work():
        nonlocal ticks
        while not release.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    with (
        patch.object(install, "gliner_runtime_installed", return_value=False),
        patch.object(install, "install_gliner_runtime", side_effect=slow_install),
    ):
        ensure = asyncio.create_task(
            cognify_config_module.ensure_extractor_runtime("gliner_demo", _config())
        )
        other = asyncio.create_task(other_work())
        await asyncio.to_thread(installing.wait, 5)
        await asyncio.sleep(0.1)
        assert not ensure.done(), "the caller must wait for the install"
        ticks_during_install = ticks
        release.set()
        await ensure
        await other

    assert ticks_during_install > 3, "the event loop was blocked during the install"
    assert install_thread and install_thread[0] != loop_thread
    assert [name for name, _ in events] == [
        "GLiNER Runtime Install Started",
        "GLiNER Runtime Install Completed",
    ]
    completed = events[1][1]
    assert completed["installed"] == ["torch", "gliner2"]
    assert completed["duration_seconds"] == 42
    assert completed["installer"] == "pip"


@pytest.mark.asyncio
async def test_a_failed_install_is_reported_by_step_and_class_only(events):
    error = install.GlinerInstallError(
        "downloading CPU-only PyTorch failed (/home/alice/.venv/bin/python -m pip ...): "
        "ERROR: https://mirror.internal.example/cpu unreachable",
        "torch",
        "pip",
    )

    def failing_install(index_url, on_start):
        on_start()
        raise error

    with (
        patch.object(install, "gliner_runtime_installed", return_value=False),
        patch.object(install, "install_gliner_runtime", side_effect=failing_install),
        pytest.raises(install.GlinerInstallError),
    ):
        await cognify_config_module.ensure_extractor_runtime(
            "gliner_demo", _config(gliner_torch_index_url="https://mirror.internal.example/cpu")
        )
    assert [name for name, _ in events] == [
        "GLiNER Runtime Install Started",
        "GLiNER Runtime Install Failed",
    ]
    failed = events[1][1]
    assert (failed["failed_step"], failed["installer"], failed["exception_type"]) == (
        "torch",
        "pip",
        "GlinerInstallError",
    )
    assert failed["torch_index"] == "custom"


@pytest.mark.asyncio
async def test_a_caller_that_only_waited_on_the_lock_sends_no_events(events):
    """Concurrent cognify calls: the waiter finds the runtime installed and stays silent."""
    with (
        patch.object(install, "gliner_runtime_installed", return_value=False),
        patch.object(install, "install_gliner_runtime", return_value=install.InstallOutcome()),
    ):
        await cognify_config_module.ensure_extractor_runtime("gliner_demo", _config())
    assert events == []


@pytest.mark.asyncio
async def test_install_events_carry_no_paths_urls_or_messages(events):
    """Privacy: only versions, platform names, step names and class names are sent."""
    import sys as _sys

    error = install.GlinerInstallError("secret /home/alice path", "gliner2", "uv")
    with (
        patch.object(install, "gliner_runtime_installed", return_value=False),
        patch.object(install, "install_gliner_runtime", side_effect=error),
        pytest.raises(install.GlinerInstallError),
    ):
        await cognify_config_module.ensure_extractor_runtime(
            "gliner_demo", _config(gliner_torch_index_url="https://mirror.internal.example/cpu")
        )
    allowed = {
        "cognee_version",
        "python_version",
        "os",
        "arch",
        "torch_index",
        "failed_step",
        "installer",
        "exception_type",
        "installed",
        "torch_version",
        "duration_seconds",
    }
    for _, properties in events:
        assert set(properties) <= allowed
        flat = repr(properties)
        for leak in ("/home/alice", "mirror.internal", "secret", _sys.prefix, _sys.executable):
            assert leak not in flat
