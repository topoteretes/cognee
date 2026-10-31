"""Journey 1: the first five minutes.

Build the wheel from this checkout, install it into a brand-new virtualenv with
nothing else, and run the README quickstart (remember, then recall) plus the
CLI entry point. This is the only test that exercises packaging, entry points,
first-run database creation and import-time side effects the way a new user
meets them.

Slow (wheel build + dependency install), so it is opt-in:
``COGNEE_JOURNEY_QUICKSTART=1 pytest -m quickstart cognee/tests/journeys``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cognee.tests.journeys import _support

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# Generous: includes cold interpreter start and first-run migrations, not install.
QUICKSTART_BUDGET_S = {"mock": 180, "llm": 600}

pytestmark = [
    pytest.mark.journey,
    pytest.mark.quickstart,
    pytest.mark.skipif(
        os.getenv("COGNEE_JOURNEY_QUICKSTART") != "1",
        reason="set COGNEE_JOURNEY_QUICKSTART=1 to build a wheel and run the quickstart in a fresh venv",
    ),
    pytest.mark.skipif(
        shutil.which("uv") is None, reason="uv is required to build and install the wheel"
    ),
]


def _venv_bin(venv: Path, name: str) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


@pytest.fixture(scope="module")
def fresh_install(tmp_path_factory) -> dict:
    """Build the wheel once and install it into an empty venv."""
    work = tmp_path_factory.mktemp("quickstart")
    dist = work / "dist"
    venv = work / "venv"

    build = _run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=REPO_ROOT)
    assert build.returncode == 0, f"uv build failed:\n{build.stdout}\n{build.stderr}"
    wheels = sorted(dist.glob("cognee-*.whl"))
    assert wheels, f"no wheel produced in {dist}: {list(dist.iterdir())}"
    wheel = wheels[-1]

    create = _run(["uv", "venv", str(venv), "--python", sys.executable])
    assert create.returncode == 0, f"uv venv failed:\n{create.stdout}\n{create.stderr}"

    python = _venv_bin(venv, "python")
    install = _run(["uv", "pip", "install", "--python", str(python), str(wheel)])
    assert install.returncode == 0, (
        f"installing the wheel failed:\n{install.stdout}\n{install.stderr[-4000:]}"
    )

    return {"venv": venv, "python": python, "wheel": wheel, "work": work}


def _clean_env(work: Path) -> dict:
    """Environment a brand-new user would have: no repo on sys.path, no secrets in mock mode."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LLM_", "EMBEDDING_", "GRAPH_DATABASE", "VECTOR_DB", "DB_", "COGNEE_"))
        and k not in ("PYTHONPATH", "VIRTUAL_ENV")
    }
    env.update(
        {
            "DATA_ROOT_DIRECTORY": str(work / "data"),
            "SYSTEM_ROOT_DIRECTORY": str(work / "system"),
            "TELEMETRY_DISABLED": "1",
            "ENV": "dev",
        }
    )
    if _support.IS_MOCK:
        env.update(
            {
                "LLM_API_KEY": "mock-key",
                "LLM_PROVIDER": "openai",
                "LLM_MODEL": "openai/gpt-5-mini",
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "openai/text-embedding-3-small",
                "EMBEDDING_DIMENSIONS": "256",
                "EMBEDDING_API_KEY": "mock-key",
                "COGNEE_SKIP_PREFLIGHT": "1",
                "COGNEE_SKIP_CONNECTION_TEST": "true",
            }
        )
    else:
        for key in (
            "LLM_API_KEY",
            "LLM_PROVIDER",
            "LLM_MODEL",
            "EMBEDDING_PROVIDER",
            "EMBEDDING_MODEL",
            "EMBEDDING_API_KEY",
            "LLM_ENDPOINT",
            "EMBEDDING_ENDPOINT",
            "EMBEDDING_DIMENSIONS",
        ):
            if os.environ.get(key):
                env[key] = os.environ[key]
    return env


def test_wheel_installs_and_imports_without_the_repo(fresh_install):
    python = fresh_install["python"]
    probe = _run(
        [
            str(python),
            "-c",
            "import cognee, json; print(json.dumps({'file': cognee.__file__, 'version': cognee.__version__}))",
        ],
        env=_clean_env(fresh_install["work"]),
        cwd=fresh_install["work"],
    )
    assert probe.returncode == 0, f"import cognee failed in the fresh venv:\n{probe.stderr[-3000:]}"
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    assert str(fresh_install["venv"]) in info["file"], (
        f"cognee imported from outside the venv: {info}"
    )
    assert info["version"], "version string is empty"


def test_cli_entry_point_is_installed(fresh_install):
    cli = _venv_bin(fresh_install["venv"], "cognee-cli")
    assert cli.exists(), f"cognee-cli entry point missing at {cli}"
    result = _run(
        [str(cli), "--version"], env=_clean_env(fresh_install["work"]), cwd=fresh_install["work"]
    )
    assert result.returncode == 0, f"cognee-cli --version failed:\n{result.stderr[-2000:]}"
    assert result.stdout.strip() or result.stderr.strip(), "cognee-cli --version printed nothing"
    helptext = _run(
        [str(cli), "--help"], env=_clean_env(fresh_install["work"]), cwd=fresh_install["work"]
    )
    assert helptext.returncode == 0
    for command in ("remember", "recall"):
        assert command in helptext.stdout, f"cognee-cli --help does not list `{command}`"


def test_readme_quickstart_remember_then_recall(fresh_install, journey_mode):
    work = fresh_install["work"]
    script_dir = work / "quickstart"
    script_dir.mkdir(exist_ok=True)
    shutil.copy(HERE / "quickstart_script.py", script_dir / "quickstart_script.py")
    shutil.copy(HERE / "mock_ai.py", script_dir / "mock_ai.py")

    env = _clean_env(work)
    started = time.monotonic()
    result = _run(
        [
            str(fresh_install["python"]),
            str(script_dir / "quickstart_script.py"),
            "--mode",
            journey_mode,
        ],
        env=env,
        cwd=script_dir,
    )
    elapsed = time.monotonic() - started

    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, (
        f"quickstart printed no report. stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-3000:]}"
    )
    report = json.loads(lines[-1])

    assert result.returncode == 0 and report.get("ok"), (
        f"quickstart failed: {json.dumps(report, indent=2)[:3000]}\nstderr:\n{result.stderr[-3000:]}"
    )
    assert report["remember_status"] == "completed"
    assert report["recall_answered"], (
        f"recall did not return the remembered fact: {report.get('recall_texts')}"
    )
    assert "Traceback" not in result.stderr, (
        f"quickstart logged a traceback:\n{result.stderr[-3000:]}"
    )
    assert elapsed < QUICKSTART_BUDGET_S[journey_mode], (
        f"quickstart took {elapsed:.0f}s, budget is {QUICKSTART_BUDGET_S[journey_mode]}s"
    )

    # First run created the local databases where the env said to.
    system_root = Path(env["SYSTEM_ROOT_DIRECTORY"])
    assert system_root.exists() and any(system_root.rglob("*")), (
        f"no system files were created under {system_root}"
    )
    data_root = Path(env["DATA_ROOT_DIRECTORY"])
    assert data_root.exists(), f"no data root created under {data_root}"
