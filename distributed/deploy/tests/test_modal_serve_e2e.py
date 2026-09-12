"""End-to-end test of the Modal serving path (distributed/deploy/modal_app.py).

Deploys an ephemeral, uniquely named Modal app, waits for the API to come up,
runs a minimal add -> cognify -> search round-trip over HTTP against the live
endpoint, and always tears the deployment down — even on failure.

This exercises the deployed FastAPI ASGI app, which is distinct from the
distributed task executor covered by .github/workflows/distributed_test.yml.
Note that the Modal image installs the published cognee package, so this
smoke-tests the released serving path rather than the local checkout.

Run locally (requires a Modal account and the cognee-secrets secret group):

    pip install modal httpx && modal setup
    python distributed/deploy/tests/test_modal_serve_e2e.py

Environment variables:
    MODAL_APP_NAME          Ephemeral app name (default: cognee-api-e2e-<run id>)
    MODAL_VOLUME_NAME       Ephemeral volume name (default: cognee-data-e2e-<run id>)
    DEFAULT_USER_EMAIL      API default-user login; must match the values in the
    DEFAULT_USER_PASSWORD   cognee-secrets Modal secret group, if set there.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
MODAL_APP_FILE = "distributed/deploy/modal_app.py"

# The deployed app scales to zero (scaledown_window=300), so the first request
# pays a container cold start on top of the image pull — poll generously.
HEALTH_DEADLINE_SECONDS = 180

WEB_ENDPOINT_PATTERN = re.compile(r"https://[A-Za-z0-9.\-]+\.modal\.run")


def deploy(app_name: str, volume_name: str) -> str:
    """Deploy the app under an ephemeral name and return its web endpoint URL.

    The URL is parsed from the deploy output rather than constructed, because
    renaming the app changes the endpoint label.
    """
    env = {**os.environ, "MODAL_APP_NAME": app_name, "MODAL_VOLUME_NAME": volume_name}
    result = subprocess.run(
        ["modal", "deploy", MODAL_APP_FILE],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    print(output)

    if result.returncode != 0:
        raise RuntimeError(f"modal deploy exited with code {result.returncode}")

    match = WEB_ENDPOINT_PATTERN.search(output)
    if match is None:
        raise RuntimeError("no *.modal.run web endpoint URL found in modal deploy output")
    return match.group(0)


def wait_for_health(client: httpx.Client, base_url: str) -> None:
    """Poll GET /health with exponential backoff until it returns 200."""
    deadline = time.monotonic() + HEALTH_DEADLINE_SECONDS
    delay = 2.0
    last_error = "no request made"

    while time.monotonic() < deadline:
        try:
            response = client.get(f"{base_url}/health", timeout=30)
            if response.status_code == 200:
                print(f"health: {response.json()}")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as error:
            last_error = repr(error)
        time.sleep(delay)
        delay = min(delay * 2, 30)

    raise RuntimeError(
        f"GET /health did not return 200 within {HEALTH_DEADLINE_SECONDS}s; "
        f"last error: {last_error}"
    )


def login(client: httpx.Client, base_url: str) -> dict:
    """Log in as the default user and return auth headers.

    The server creates the default user from DEFAULT_USER_EMAIL /
    DEFAULT_USER_PASSWORD (set via the cognee-secrets group, with cognee's
    built-in defaults otherwise). If login fails we continue without a token,
    which still works for deployments running with authentication disabled.
    """
    response = client.post(
        f"{base_url}/api/v1/auth/login",
        data={
            "username": os.getenv("DEFAULT_USER_EMAIL", "default_user@example.com"),
            "password": os.getenv("DEFAULT_USER_PASSWORD", "default_password"),
        },
        timeout=60,
    )
    if response.status_code == 200:
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    print(f"login failed ({response.status_code}); continuing unauthenticated", file=sys.stderr)
    return {}


def run_golden_flow(base_url: str, client: httpx.Client) -> None:
    """Minimal add -> cognify -> search round-trip against the live endpoint.

    Self-contained stand-in for the shared deployment-test harness (issue
    #3358); swap this for the harness golden_flow() once it lands. The input
    text is intentionally tiny to bound LLM cost and runtime.
    """
    dataset = "modal_serve_e2e"
    text = "Cognee turns documents into an AI memory layer. Modal serves the cognee API."

    response = client.post(
        f"{base_url}/api/v1/add",
        files={"data": ("golden.txt", text.encode(), "text/plain")},
        data={"datasetName": dataset},
        timeout=300,
    )
    assert response.status_code == 200, f"add failed: {response.status_code} {response.text[:500]}"
    print(f"add: {response.json().get('status')}")

    response = client.post(
        f"{base_url}/api/v1/cognify",
        json={"datasets": [dataset], "run_in_background": False},
        timeout=1200,
    )
    assert response.status_code == 200, (
        f"cognify failed: {response.status_code} {response.text[:500]}"
    )
    print("cognify: completed")

    response = client.post(
        f"{base_url}/api/v1/search",
        json={
            "search_type": "GRAPH_COMPLETION",
            "datasets": [dataset],
            "query": "What does cognee do?",
        },
        timeout=300,
    )
    assert response.status_code == 200, (
        f"search failed: {response.status_code} {response.text[:500]}"
    )
    results = response.json()
    assert isinstance(results, list) and len(results) > 0, f"search returned no results: {results}"
    print(f"search: {len(results)} result(s)")


def dump_app_logs(app_name: str) -> None:
    """Print recent app logs to aid debugging. `modal app logs` streams
    indefinitely, so cut it off after a short timeout."""
    try:
        subprocess.run(["modal", "app", "logs", app_name], timeout=20)
    except subprocess.TimeoutExpired:
        pass
    except Exception as error:
        print(f"could not fetch app logs: {error!r}", file=sys.stderr)


def teardown(app_name: str, volume_name: str) -> None:
    """Best-effort removal of the ephemeral app and volume; never raises."""
    for command in (
        ["modal", "app", "stop", app_name, "--yes"],
        ["modal", "volume", "delete", volume_name, "--yes", "--allow-missing"],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        outcome = (
            "ok"
            if result.returncode == 0
            else f"failed ({result.returncode}): {result.stderr.strip()[:300]}"
        )
        print(f"teardown: {' '.join(command)} -> {outcome}")


def main() -> int:
    run_id = os.getenv("GITHUB_RUN_ID", str(int(time.time())))
    app_name = os.getenv("MODAL_APP_NAME", f"cognee-api-e2e-{run_id}")
    volume_name = os.getenv("MODAL_VOLUME_NAME", f"cognee-data-e2e-{run_id}")

    # Teardown deletes the volume, so never run against the production names.
    if app_name == "cognee-api" or volume_name == "cognee-data":
        print("refusing to run against production app/volume names", file=sys.stderr)
        return 2

    try:
        base_url = deploy(app_name, volume_name)
        print(f"deployed {app_name} at {base_url}")

        with httpx.Client() as client:
            wait_for_health(client, base_url)
            client.headers.update(login(client, base_url))
            run_golden_flow(base_url, client)

        print("Modal serve e2e passed")
        return 0
    except Exception:
        dump_app_logs(app_name)
        raise
    finally:
        teardown(app_name, volume_name)


if __name__ == "__main__":
    sys.exit(main())
