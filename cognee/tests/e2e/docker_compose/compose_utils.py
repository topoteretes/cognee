"""Thin wrappers around the ``docker compose`` CLI and HTTP health polling.

These helpers let the e2e test orchestrate the parts of the stack it needs:
waiting on healthchecks (replacing the old ``sleep 30``), recreating a service
for the persistence check, and reading service logs for the traceback scan.
"""

from __future__ import annotations

import subprocess
import time
from typing import List, Optional

import requests

from config import CONFIG


class ServiceNotHealthy(AssertionError):
    """Raised when a service fails to become healthy within the timeout."""


def _compose_base_cmd() -> List[str]:
    cmd = ["docker", "compose", "-f", CONFIG.compose_file]
    for profile in CONFIG.compose_profiles:
        cmd += ["--profile", profile]
    return cmd


def compose(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a ``docker compose`` subcommand against the configured stack."""
    cmd = _compose_base_cmd() + list(args)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def service_logs(service: Optional[str] = None) -> str:
    """Return logs for a single service (or the whole stack when omitted)."""
    args = ["logs", "--no-color"]
    if service:
        args.append(service)
    result = compose(*args, check=False, capture=True)
    # `docker compose logs` interleaves stdout/stderr; concatenate both.
    return (result.stdout or "") + (result.stderr or "")


def recreate_service(service: str) -> None:
    """Force-recreate a single service container.

    Unlike ``restart`` (which reuses the same container and so keeps its
    writable layer), ``up --force-recreate`` throws the container away and
    builds a fresh one. Named volumes survive; anything written only to the
    container layer does not. This is what makes the Postgres persistence
    check meaningful — without the named volume the data would be gone.
    """
    compose("up", "-d", "--force-recreate", "--no-deps", service)


def wait_for_http_ok(
    url: str,
    *,
    timeout: Optional[float] = None,
    poll_interval: Optional[float] = None,
    name: Optional[str] = None,
    expect_status: tuple = (200,),
) -> requests.Response:
    """Poll ``url`` until it returns an expected status or the timeout elapses."""
    timeout = CONFIG.startup_timeout if timeout is None else timeout
    poll_interval = CONFIG.poll_interval if poll_interval is None else poll_interval
    label = name or url

    deadline = time.monotonic() + timeout
    last_error: Optional[str] = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code in expect_status:
                return response
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = repr(exc)
        time.sleep(poll_interval)

    raise ServiceNotHealthy(
        f"Service '{label}' did not become healthy within {timeout:.0f}s "
        f"(last error: {last_error})."
    )
