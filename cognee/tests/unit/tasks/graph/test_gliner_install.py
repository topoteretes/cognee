"""The GLiNER runtime install: torch (CPU build) only when missing, then the gliner extra."""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from cognee.tasks.graph.gliner_demo import install

COGNEE_REQUIRES = [
    "filelock>=3.12.0,<4.0.0",
    'numpy>=1.26; python_version >= "3.10"',
    'gliner2[local]<3,>=2.0.0; extra == "gliner"',
    'protobuf>=5.29.6; extra == "gliner"',
    'boto3>=1.26; extra == "aws"',
]


def test_gliner_requirements_are_the_gliner_extra_only():
    with patch("importlib.metadata.requires", return_value=COGNEE_REQUIRES):
        assert install.gliner_requirements() == ["gliner2[local]<3,>=2.0.0", "protobuf>=5.29.6"]


def test_gliner_requirements_raise_without_the_extra():
    with (
        patch("importlib.metadata.requires", return_value=COGNEE_REQUIRES[:2]),
        pytest.raises(install.GlinerInstallError, match="no `gliner` extra"),
    ):
        install.gliner_requirements()


def test_installer_prefers_pip_then_uv_then_raises():
    with patch("importlib.util.find_spec", return_value=object()):
        assert install.installer_command() == [sys.executable, "-m", "pip", "install"]
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("shutil.which", return_value="/bin/uv"),
    ):
        assert install.installer_command() == [
            "/bin/uv",
            "pip",
            "install",
            "--python",
            sys.executable,
        ]
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("shutil.which", return_value=None),
        pytest.raises(install.GlinerInstallError, match="neither pip nor uv"),
    ):
        install.installer_command()


def _install(tmp_path, present, installed_after=True):
    """Run install_gliner_runtime with the environment faked.

    ``present`` is the set of modules importable before the install; returns the
    steps run as ``(command, step, constraints)``, the constraints file read while
    it existed.
    """
    runs = []

    def run(command, step):
        constraints = None
        if "-c" in command:
            constraints = Path(command[command.index("-c") + 1]).read_text().split()
        runs.append((command, step, constraints))

    def runtime_installed():
        return installed_after if runs else {"gliner2", "torch"} <= present

    with (
        patch.object(sys, "prefix", str(tmp_path)),
        patch.object(install, "_installed", side_effect=lambda module: module in present),
        patch.object(install, "gliner_runtime_installed", side_effect=runtime_installed),
        patch.object(install, "installer_command", return_value=["pip", "install"]),
        patch.object(install, "_run", side_effect=run),
        patch.object(install, "installed_pins", return_value=["numpy==2.2.0"]),
        patch.object(install, "gliner_requirements", return_value=["gliner2[local]<3,>=2.0.0"]),
        patch("importlib.metadata.version", return_value="2.9.1+cpu"),
    ):
        install.install_gliner_runtime("https://mirror.example/cpu")
    return runs


def test_missing_torch_comes_from_the_cpu_index_then_the_extra_with_installed_pins(tmp_path):
    (torch_step, gliner_step) = _install(tmp_path, present=set())
    assert torch_step[0] == [
        "pip",
        "install",
        "--index-url",
        "https://mirror.example/cpu",
        "torch>=2.1,<3",
    ]
    assert "step 1 of 2" in torch_step[1]
    command, step, constraints = gliner_step
    assert command[:3] == ["pip", "install", "-c"]
    assert command[-1] == "gliner2[local]<3,>=2.0.0"
    assert constraints == ["numpy==2.2.0"]
    assert "step 2 of 2" in step


def test_an_installed_torch_cpu_or_gpu_is_kept(tmp_path):
    runs = _install(tmp_path, present={"torch"})
    assert [step for _, step, _ in runs] == [
        "installing gliner2 and its tokenizer dependencies (step 2 of 2)"
    ]


def test_install_is_a_no_op_when_the_runtime_is_present(tmp_path):
    assert _install(tmp_path, present={"torch", "gliner2"}) == []


def test_install_raises_when_the_runtime_is_still_not_importable(tmp_path):
    with pytest.raises(install.GlinerInstallError, match="still cannot import"):
        _install(tmp_path, present=set(), installed_after=False)


def test_run_relays_progress_lines_and_raises_with_the_output_tail(caplog):
    script = (
        "print('Collecting torch'); print('Requirement already satisfied: x');"
        "print('Successfully installed torch-2.9.1'); raise SystemExit(0)"
    )
    with caplog.at_level(logging.INFO):
        install._run([sys.executable, "-c", script], "downloading CPU-only PyTorch")
    logged = " | ".join(record.getMessage() for record in caplog.records)
    assert "GLiNER runtime: Collecting torch" in logged
    assert "GLiNER runtime: Successfully installed torch-2.9.1" in logged
    assert "Requirement already satisfied" not in logged
    assert "GLiNER runtime: done downloading CPU-only PyTorch in" in logged

    failing = "print('ERROR: No matching distribution found for torch'); raise SystemExit(1)"
    with pytest.raises(install.GlinerInstallError, match="No matching distribution"):
        install._run([sys.executable, "-c", failing], "downloading CPU-only PyTorch")


def test_run_reports_elapsed_time_while_a_step_is_slow(caplog):
    with patch.object(install, "HEARTBEAT_SECONDS", 0.05), caplog.at_level(logging.INFO):
        install._run([sys.executable, "-c", "import time; time.sleep(0.3)"], "downloading")
    assert any("still downloading" in record.getMessage() for record in caplog.records)
