"""Top-level serve() orchestrator — connects the SDK to a remote or local Cognee instance."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import CloudCredentials

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve")


async def serve(url: str | None = None, api_key: str | None = None) -> CloudClient:
    """Connect the local Cognee SDK to a remote or local Cognee instance.

    Connects to the instance at ``url`` with ``api_key``::

        await cognee.serve(url="http://localhost:8000")
        await cognee.serve(url="https://my-instance.cognee.ai", api_key="ck_...")

    Both can also come from ``COGNEE_SERVICE_URL`` / ``COGNEE_API_KEY``. A
    successful connection is saved, and a later bare ``serve()`` reconnects to
    the same instance without arguments when that connection used an API key.
    A connection without a key (a local instance with auth off) is not reused:
    pass its ``url`` again.

    Once connected, all operations (remember, recall, improve, forget,
    visualize) route to the instance instead of running locally.

    Args:
        url: URL of the Cognee instance. Falls back to ``COGNEE_SERVICE_URL``,
            then to the saved connection.
        api_key: API key for the instance. Falls back to ``COGNEE_API_KEY``,
            then to the saved connection.

    Returns:
        CloudClient connected to the instance.

    Raises:
        CogneeConfigurationError: When called with no ``url``, no environment
            variables and no usable saved connection — there is nothing to
            connect with, and the error explains the options. Also raised when
            the instance is reachable but rejects the API key (401/403 on an
            authenticated probe) — connecting would otherwise appear to succeed
            and fail on the first operation.
    """
    service_url = url or os.getenv("COGNEE_SERVICE_URL")
    resolved_api_key = api_key or os.getenv("COGNEE_API_KEY", "")

    if service_url:
        return await _serve_direct(service_url, resolved_api_key)

    from cognee.api.v1.serve.credentials import get_credentials_path, load_credentials
    from cognee.exceptions import CogneeConfigurationError

    creds = load_credentials()
    # Only a keyed connection is reused: the key is what tells a later bare
    # serve() that the instance at the saved URL is still the one it was.
    if creds and creds.service_url and creds.api_key:
        return await _serve_saved(creds)

    raise CogneeConfigurationError(
        message=(
            "cognee.serve() was called without connection details and none were found "
            f"(no saved connection at {get_credentials_path()}, no environment variables).\n\n"
            "To connect to a Cognee instance, pass its URL and API key:\n"
            '    await cognee.serve(url="http://localhost:8000")\n'
            '    await cognee.serve(url="https://your-instance", api_key="...")\n'
            "or set the COGNEE_SERVICE_URL and COGNEE_API_KEY environment variables."
        ),
        name="ServeConfigurationError",
    )


async def _serve_direct(service_url: str, api_key: str = "") -> CloudClient:
    """Connect to the instance at ``service_url`` and save the connection."""
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import CloudCredentials, save_credentials
    from cognee.api.v1.serve.state import set_remote_client

    service_url = service_url.rstrip("/")
    client = CloudClient(service_url, api_key)

    health_ok = await client._health_check()
    if not health_ok:
        logger.warning("Instance at %s did not respond to health check", service_url)

    # /health is unauthenticated, so it cannot catch a rejected API key —
    # without this probe a bad key still prints "Connected" and then every
    # operation fails with 401. Only a definitive rejection is fatal; an
    # unreachable instance keeps the warn-and-continue behavior above.
    auth_status = await client._auth_check()
    if auth_status in (401, 403):
        from cognee.exceptions import CogneeConfigurationError

        await client.close()
        if api_key:
            detail = "the API key was rejected — check that it belongs to this instance."
        else:
            detail = (
                "it requires authentication and no API key was given. Pass one via "
                'cognee.serve(url=..., api_key="...") or the COGNEE_API_KEY environment '
                "variable. On a self-hosted server, mint a key by logging in and calling "
                "POST /api/v1/auth/api-keys (the key is only revealed in the creation "
                "response)."
            )
        raise CogneeConfigurationError(
            message=f"Connected to {service_url}, but {detail}",
            name="ServeAuthenticationError",
        )

    # The tenant id links this install's telemetry to the tenant it connected
    # to (CLO-804); telemetry reads it from the client while connected.
    client.tenant_id = await client._get_tenant_id()
    # Save so subsequent serve() calls reconnect without args
    save_credentials(
        CloudCredentials(service_url=service_url, api_key=api_key, tenant_id=client.tenant_id)
    )

    set_remote_client(client)
    mode = "local" if "localhost" in service_url or "127.0.0.1" in service_url else "remote"
    print(f"  Connected to Cognee ({mode}) at {service_url}")
    return client


def _save_tenant_id(creds: CloudCredentials, tenant_id: str) -> None:
    """Record the tenant on a saved connection; optional, so never fails serve()."""
    from cognee.api.v1.serve.credentials import CloudCredentials, save_credentials

    try:
        save_credentials(
            CloudCredentials(
                service_url=creds.service_url, api_key=creds.api_key, tenant_id=tenant_id
            )
        )
    except Exception:
        logger.debug("Could not save the tenant id with the saved connection", exc_info=True)


async def _serve_saved(creds: CloudCredentials) -> CloudClient:
    """Reconnect to the saved instance, or explain why that no longer works."""
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import get_credentials_path
    from cognee.api.v1.serve.state import set_remote_client
    from cognee.exceptions import CogneeConfigurationError

    client = CloudClient(creds.service_url, creds.api_key)
    try:
        # The saved API key must actually authenticate, not just reach the
        # instance — a stale key on a healthy instance would otherwise
        # "connect" and fail on the first operation.
        if await client._health_check() and await client._auth_check() not in (401, 403):
            logger.info("Using saved connection to %s", creds.service_url)
            client.tenant_id = creds.tenant_id
            if not client.tenant_id:
                # Connections saved before the tenant id was recorded.
                client.tenant_id = await client._get_tenant_id()
                if client.tenant_id:
                    _save_tenant_id(creds, client.tenant_id)
            set_remote_client(client)
            print(f"  Connected to Cognee at {creds.service_url}")
            return client
    except Exception as e:
        logger.warning("Saved instance check failed: %s", e, exc_info=True)
    await client.close()

    raise CogneeConfigurationError(
        message=(
            f"cognee.serve() found a saved connection ({get_credentials_path()}), but the "
            f"saved instance at {creds.service_url} is unreachable or its API key no longer "
            "works.\n\n"
            "To connect to a Cognee instance, pass its URL and API key:\n"
            '    await cognee.serve(url="https://your-instance", api_key="...")\n'
            "or set the COGNEE_SERVICE_URL and COGNEE_API_KEY environment variables.\n\n"
            "If the saved connection is stale, delete the file above and connect again."
        ),
        name="ServeConfigurationError",
    )
