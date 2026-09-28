"""Entrypoint argv assembly tests (#5145).

The entrypoint builds the cognee-mcp argv from env-derived defaults
(TRANSPORT_MODE, HTTP_PORT, API_URL/API_TOKEN) plus the runtime args passed
to `docker run ...`. Runtime args are appended last and argparse keeps the
last occurrence of a flag, so a runtime `--transport http` must override the
env default (previously the defaults were appended after the user args and
silently replaced them).

Runs the real entrypoint.sh with a fake `cognee-mcp` on PATH that dumps its
argv, so the assertions cover the actual exec, not a reimplementation.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

COGNEE_MCP_DIR = Path(__file__).resolve().parents[1]
ENTRYPOINT = COGNEE_MCP_DIR / "entrypoint.sh"


def _find_bash() -> str:
    """Pick a real bash for the entrypoint subprocess.

    On Windows, `bash` on PATH may resolve to C:\\Windows\\System32\\bash.exe
    — the WSL launcher — which cannot run host-side scripts. Prefer any other
    bash on PATH (Git Bash / MSYS2), then the usual Git for Windows location.
    """
    from_path = shutil.which("bash")
    if from_path and "system32" not in from_path.lower():
        return from_path
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidate = Path(program_files) / "Git" / "bin" / "bash.exe"
    if candidate.is_file():
        return str(candidate)
    return from_path or "bash"


@pytest.fixture
def sandbox(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    out_file = tmp_path / "argv.json"

    fake = bin_dir / "cognee-mcp"
    fake.write_text(
        "#!/bin/bash\n"
        f'python -c "import sys, json; json.dump(sys.argv[1:], open({str(out_file.as_posix())!r}, \'w\'))" "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    env = os.environ.copy()
    # Windows-style bin dir first: MSYS2 converts the whole PATH on startup,
    # so the fake cognee-mcp resolves inside the entrypoint's bash.
    env["PATH"] = f"{bin_dir!s}{os.pathsep}{env['PATH']}"
    env["MEMORY_DATA_DIR"] = str(data_dir)
    for key in (
        "TRANSPORT_MODE",
        "HTTP_PORT",
        "EXTRAS",
        "API_URL",
        "API_TOKEN",
        "DEBUG",
        "ENV",
        "ENVIRONMENT",
    ):
        env.pop(key, None)

    yield env, out_file
    shutil.rmtree(tmp_path, ignore_errors=True)


def _run_entrypoint(sandbox, user_args: list[str], extra_env: dict[str, str]) -> list[str]:
    env, out_file = sandbox
    env.update(extra_env)
    stdout_file = out_file.parent / "entrypoint.stdout"
    stderr_file = out_file.parent / "entrypoint.stderr"
    with (
        open(stdout_file, "w", encoding="utf-8") as out,
        open(stderr_file, "w", encoding="utf-8") as err,
    ):
        proc = subprocess.run(
            [_find_bash(), str(ENTRYPOINT), *user_args],
            env=env,
            check=False,
            timeout=120,
            stdout=out,
            stderr=err,
        )
    if proc.returncode != 0:
        stderr_text = stderr_file.read_text(encoding="utf-8", errors="replace")
        stdout_text = stdout_file.read_text(encoding="utf-8", errors="replace")
        raise AssertionError(
            f"entrypoint exited {proc.returncode}\nstderr:\n{stderr_text}\nstdout:\n{stdout_text}"
        )
    return json.loads(out_file.read_text(encoding="utf-8"))


def last_value_for(argv: list[str], flag: str) -> str | None:
    """argparse keeps the last occurrence — the value that actually applies."""
    value = None
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            value = argv[i + 1]
    return value


def test_runtime_transport_http_wins_over_unset_env(sandbox):
    argv = _run_entrypoint(sandbox, ["--transport", "http"], {})
    # the runtime choice is the one argparse keeps (last occurrence) ...
    assert last_value_for(argv, "--transport") == "http"
    # ... and the http bind defaults follow the effective transport
    assert last_value_for(argv, "--host") == "0.0.0.0"
    assert last_value_for(argv, "--port") == "8000"


def test_runtime_args_win_over_env_defaults(sandbox):
    argv = _run_entrypoint(
        sandbox,
        ["--transport", "http", "--port", "9000"],
        {"TRANSPORT_MODE": "http"},
    )
    assert last_value_for(argv, "--transport") == "http"
    assert last_value_for(argv, "--port") == "9000"


def test_env_defaults_unchanged_when_no_runtime_args(sandbox):
    argv = _run_entrypoint(sandbox, [], {"TRANSPORT_MODE": "http"})
    assert argv == ["--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
