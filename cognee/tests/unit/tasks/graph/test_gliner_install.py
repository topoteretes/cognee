"""The GLiNER runtime install: torch (CPU build) only when missing, then the gliner extra."""

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

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
        patch.object(
            install, "gliner_extra", return_value=("torch<3,>=2.1", ["gliner2[local]<3,>=2.0.0"])
        ),
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
        "torch<3,>=2.1",
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


def test_a_successful_install_suggests_the_extra(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _install(tmp_path, present=set())
    logged = " | ".join(record.getMessage() for record in caplog.records)
    assert (
        'Tip: if you keep using GLiNER, install cognee with the extra (pip install "cognee[gliner]")'
        in logged
    )


def test_an_unwritable_environment_raises_with_the_extra_as_the_fix(tmp_path):
    read_only = tmp_path / "prefix"
    read_only.mkdir(mode=0o555)
    try:
        with (
            patch.object(sys, "prefix", str(read_only)),
            patch.object(install, "_run", side_effect=AssertionError("must not install")),
            pytest.raises(install.GlinerInstallError, match="is not writable") as raised,
        ):
            install.install_gliner_runtime("https://mirror.example/cpu")
    finally:
        read_only.chmod(0o755)
    assert "cognee[gliner]" in raised.value.remediation


def test_the_constraints_file_is_removed_even_when_writing_it_fails(tmp_path):
    with (
        patch.object(tempfile, "tempdir", str(tmp_path)),
        patch.object(install, "installed_pins", side_effect=RuntimeError("bad metadata")),
        pytest.raises(RuntimeError, match="bad metadata"),
    ):
        install._install_rest(["pip", "install"], ["gliner2"])
    assert list(tmp_path.iterdir()) == []


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
