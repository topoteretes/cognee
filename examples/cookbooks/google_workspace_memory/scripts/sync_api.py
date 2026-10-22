"""Sync Calendar, Gmail and Drive through a running cognee server's Google integrations.

The ingest_* scripts read Google from this machine with your own OAuth client. With --api the
cookbook asks a cognee server to do it instead: the server holds the OAuth clients, each user
connects their Google account once in the browser, and the server syncs each app into that
account's own dataset. Use it when cognee runs as a service for several people.

For each app this selects what to sync (your primary calendar, the INBOX label, the folder
GOOGLE_DRIVE_FOLDER_ID), starts a sync and waits for it to finish. It returns the dataset
ids the server synced into, which scripts/ask.py --api then asks across.

The server needs, per app (prefixes GOOGLE_CALENDAR_, GOOGLE_GMAIL_, GOOGLE_DRIVE_):
CLIENT_ID, CLIENT_SECRET (a Google OAuth client of type Web application), REDIRECT_URI
(this server's /api/v1/integrations/<provider>/callback, also listed on the client),
STATE_SECRET and FRONTEND_BASE_URL; plus INTEGRATION_CREDENTIALS_KEYS and
INTEGRATION_CREDENTIALS_ACTIVE_KEY_ID to encrypt the stored tokens. See .env.template.

Connecting an app happens in the browser, never here: the consent must finish in the same
browser that started it, because the server checks a cookie it set there. Connect Drive and
Gmail on the cognee UI's Integrations page. Calendar has no card there yet: in the same
browser, open the server's /docs, call POST /api/v1/integrations/google_calendar/authorize
and open the authorizeUrl it returns.

COGNEE_API_TOKEN holds your bearer token for the server; COGNEE_API_URL its address
(default http://localhost:8000).

Run alone: uv run python examples/cookbooks/google_workspace_memory/scripts/sync_api.py [--no-calendar] [--no-gmail] [--no-drive]
"""

import argparse
import asyncio
import os
import time

import aiohttp

DEFAULT_API_URL = "http://localhost:8000"
PROVIDERS = {"calendar": "google_calendar", "gmail": "gmail", "drive": "google_drive"}
CONNECT_HINT = {
    "google_calendar": "in this browser, call POST /api/v1/integrations/google_calendar/"
    "authorize from the server's /docs and open the authorizeUrl it returns",
    "gmail": "connect Gmail on the cognee UI's Integrations page",
    "google_drive": "connect Google Drive on the cognee UI's Integrations page",
}
POLL_SECONDS = 5
WAIT_SECONDS = 60 * 60  # a first sync of a large calendar or inbox can take a while


def api_url() -> str:
    return os.environ.get("COGNEE_API_URL", DEFAULT_API_URL).rstrip("/")


def _session() -> aiohttp.ClientSession:
    token = os.environ.get("COGNEE_API_TOKEN", "")
    return aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {token}"}, raise_for_status=True
    )


def _base(provider: str) -> str:
    return f"{api_url()}/api/v1/integrations/{provider}"


async def _connection(session: aiohttp.ClientSession, provider: str) -> dict:
    async with session.get(f"{_base(provider)}/connection") as response:
        return await response.json()


async def missing_connections(apps: list[str]) -> list[str]:
    """One line per app the server has not connected yet. Empty when all are connected."""
    missing = []
    try:
        async with _session() as session:
            for app in apps:
                provider = PROVIDERS[app]
                if not (await _connection(session, provider))["connected"]:
                    missing.append(
                        f"{provider} is not connected on {api_url()}: {CONNECT_HINT[provider]}."
                    )
    except aiohttp.ClientResponseError as error:
        missing.append(
            f"{api_url()} answered {error.status} {error.message}; check COGNEE_API_TOKEN."
        )
    except aiohttp.ClientError as error:
        missing.append(f"Could not reach the cognee server at {api_url()} ({error}).")
    return missing


async def _resource_ids(session: aiohttp.ClientSession, provider: str) -> list[str]:
    """What the cookbook syncs: the primary calendar, the inbox, the Drive folder."""
    if provider == "google_drive":
        # The server accepts any folder id; its picker only lists My Drive and shared drives.
        return [os.environ["GOOGLE_DRIVE_FOLDER_ID"]]
    async with session.get(f"{_base(provider)}/resources") as response:
        resources = (await response.json())["resources"]
    if provider == "gmail":
        wanted = [resource["id"] for resource in resources if resource["id"] == "INBOX"]
    else:
        wanted = [resource["id"] for resource in resources if resource["attributes"].get("primary")]
    if not wanted:
        raise SystemExit(
            f"[sync_api] {provider} lists no {'INBOX' if provider == 'gmail' else 'primary calendar'}."
        )
    return wanted


async def _sync(session: aiohttp.ClientSession, provider: str) -> str:
    """Select, start one sync and wait for it. Returns the dataset id it synced into."""
    before = await _connection(session, provider)
    if not before["connected"]:
        raise SystemExit(
            f"[sync_api] MISSING: {provider} is not connected: {CONNECT_HINT[provider]}."
        )
    resource_ids = await _resource_ids(session, provider)
    async with session.put(f"{_base(provider)}/resources", json={"resourceIds": resource_ids}):
        pass
    async with session.post(f"{_base(provider)}/sync") as response:
        accepted = (await response.json())["accepted"]
    print(
        f"[sync_api] {provider}: {'started' if accepted else 'a sync is already running'} "
        f"for {', '.join(resource_ids)}; waiting for it to finish…",
        flush=True,
    )
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        await asyncio.sleep(POLL_SECONDS)
        state = await _connection(session, provider)
        # A new stamp means this sync (or the one already running) finished.
        if state.get("syncStatus") != "syncing" and state.get("lastSyncedAt") != before.get(
            "lastSyncedAt"
        ):
            break
        if time.monotonic() > deadline:
            raise SystemExit(
                f"[sync_api] {provider} is still syncing after {WAIT_SECONDS // 60} minutes; "
                "run again later to ask once it is done."
            )
    counts = state.get("syncCounts") or {}
    print(
        f"[sync_api] {provider}: sync {state.get('syncStatus') or 'finished'}, "
        f"{counts.get('scanned', 0)} read, {state.get('storedItems') or 0} items in the dataset"
    )
    if not state.get("datasetId"):
        raise SystemExit(f"[sync_api] {provider} synced, but the server reports no dataset.")
    return state["datasetId"]


async def sync_api(apps: list[str]) -> list[str]:
    """Sync each app on the server, one after another. Returns their dataset ids."""
    async with _session() as session:
        return [await _sync(session, PROVIDERS[app]) for app in apps]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for app in PROVIDERS:
        parser.add_argument(f"--no-{app}", action="store_true", help=f"skip {app}")
    args = parser.parse_args()
    chosen = [app for app in PROVIDERS if not getattr(args, f"no_{app}")]
    print(f"[sync_api] Dataset ids: {', '.join(asyncio.run(sync_api(chosen)))}")
