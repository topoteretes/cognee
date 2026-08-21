"""Top-level serve() orchestrator — connects the SDK to Cognee Cloud or a local instance."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cognee.api.v1.serve.cloud_client import CloudClient

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve")


async def serve(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    management_url: Optional[str] = None,
    auth0_domain: Optional[str] = None,
    auth0_client_id: Optional[str] = None,
    auth0_audience: Optional[str] = None,
    bootstrap_auth: Optional[bool] = None,
    persist_credentials: bool = True,
) -> CloudClient:
    """Connect the local Cognee SDK to a remote or local Cognee instance.

    Two modes:

    **Local / direct mode** — when ``url`` is provided (with optional
    ``api_key``), connects directly to that instance. No Auth0, no
    Management API. Use this to connect to a local Cognee backend or
    any instance where you already have the URL and credentials::

        await cognee.serve(url="http://localhost:8000")
        await cognee.serve(url="https://my-instance.cognee.ai", api_key="ck_...")

    **Cloud mode** — when ``url`` is not provided, runs the full Auth0
    Device Code Flow, discovers the tenant via the Management API, and
    connects to the cloud instance automatically::

        await cognee.serve()

    In both modes, all operations (remember, recall, improve, forget,
    visualize) route to the connected instance instead of running locally.

    Args:
        url: Direct URL of a Cognee instance. Skips Auth0 and tenant
            discovery. Can also be set via the ``COGNEE_SERVICE_URL`` env
            var (or ``COGNEE_BASE_URL``, the name the agent integrations
            use).
        api_key: API key for authentication. Used with ``url`` for direct
            connections, or via ``COGNEE_API_KEY`` env var.
        management_url: Override the Management API URL (cloud mode only).
        auth0_domain: Override the Auth0 domain (cloud mode only).
        auth0_client_id: Override the Auth0 Device Code client ID.
        auth0_audience: Override the Auth0 API audience.
        bootstrap_auth: Direct mode only — whether a missing ``api_key``
            may be minted by logging in to the instance. Default
            (``None``): allowed for loopback/private hosts only, since
            the login sends user credentials to the host. Pass ``True``
            to allow it for a remote instance you control (or set
            ``COGNEE_AUTH_BOOTSTRAP=true``); ``False`` disables it
            entirely.
        persist_credentials: Direct mode only — whether this call may
            write to ``~/.cognee/cloud_credentials.json``. Default
            ``True`` (unchanged behavior: keys are saved so later
            ``serve()`` calls reconnect without arguments). Integrations
            that manage their own keys pass ``False`` so serve stays a
            pure pass-through and never duplicates their credentials.

    Returns:
        CloudClient connected to the instance.
    """
    # Resolve URL from arg or env. COGNEE_SERVICE_URL is serve()'s own
    # variable; COGNEE_BASE_URL is what the agent integrations already
    # export, accepted as an alias so one .env serves both.
    service_url = url or os.getenv("COGNEE_SERVICE_URL") or os.getenv("COGNEE_BASE_URL")
    resolved_api_key = api_key or os.getenv("COGNEE_API_KEY", "")

    if service_url:
        return await _serve_direct(
            service_url,
            resolved_api_key,
            bootstrap_auth=bootstrap_auth,
            persist_credentials=persist_credentials,
        )

    return await _serve_cloud(
        management_url=management_url,
        auth0_domain=auth0_domain,
        auth0_client_id=auth0_client_id,
        auth0_audience=auth0_audience,
    )


async def _serve_direct(
    service_url: str,
    api_key: str = "",
    bootstrap_auth: Optional[bool] = None,
    persist_credentials: bool = True,
) -> CloudClient:
    """Connect directly to a Cognee instance — no Auth0, no Management API.

    Without an ``api_key``, one is resolved instead of connecting
    unauthenticated: a key previously saved for this URL is reused, else
    a key is minted by logging in (``COGNEE_USER_EMAIL``/
    ``COGNEE_USER_PASSWORD``, defaulting to the server's default user)
    and persisted for subsequent connects. The login sends credentials
    to the host, so minting is gated to loopback/private addresses
    unless explicitly opted in (``bootstrap_auth=True`` /
    ``COGNEE_AUTH_BOOTSTRAP=true``). Only when no key can be resolved
    does the connection proceed keyless — valid for servers running
    with authentication off.
    """
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import CloudCredentials, load_credentials, save_credentials
    from cognee.api.v1.serve.exceptions import CogneeAPIError
    from cognee.api.v1.serve.local_auth import bootstrap_allowed, login_and_mint_api_key
    from cognee.api.v1.serve.state import set_remote_client

    service_url = service_url.rstrip("/")

    key_from_cache = False
    if not api_key:
        saved = load_credentials(service_url)
        if saved and saved.api_key:
            logger.info("Reusing saved API key for %s", service_url)
            api_key = saved.api_key
            key_from_cache = True
        elif not bootstrap_allowed(service_url, bootstrap_auth):
            logger.warning(
                "No API key for %s and auth bootstrap is disabled for non-private hosts; "
                "connecting without one. Pass api_key (or set COGNEE_API_KEY), or opt in "
                "with bootstrap_auth=True / COGNEE_AUTH_BOOTSTRAP=true to mint a key by "
                "logging in to this instance.",
                service_url,
            )
        else:
            try:
                api_key = await login_and_mint_api_key(service_url)
                logger.info("Minted API key for %s via default-user login", service_url)
            except CogneeAPIError as error:
                logger.warning(
                    "Could not obtain an API key for %s (%s); connecting without one — "
                    "this only works when the server runs with authentication off.",
                    service_url,
                    error,
                )

    client = CloudClient(service_url, api_key)

    # A key that was valid when saved can be revoked server-side. When the
    # bootstrap gate allows minting for this host, let the client replace a
    # rejected cached key once instead of failing every call with 401.
    if key_from_cache and bootstrap_allowed(service_url, bootstrap_auth):

        async def _refresh_api_key() -> str:
            new_key = await login_and_mint_api_key(service_url)
            if persist_credentials:
                save_credentials(
                    CloudCredentials(
                        access_token="",
                        service_url=service_url,
                        api_key=new_key,
                        email="local",
                    )
                )
            logger.info("Replaced rejected API key for %s", service_url)
            return new_key

        client.refresh_api_key = _refresh_api_key

    health_ok = await client._health_check()
    if not health_ok:
        logger.warning("Instance at %s did not respond to health check", service_url)

    # Save so subsequent serve() calls reconnect without args. Callers that
    # manage their own keys (the agent integrations) disable this so serve
    # never duplicates credentials into its store.
    if persist_credentials:
        save_credentials(
            CloudCredentials(
                access_token="",
                service_url=service_url,
                api_key=api_key,
                email="local",
            )
        )

    set_remote_client(client)
    mode = "local" if "localhost" in service_url or "127.0.0.1" in service_url else "remote"
    print(f"  Connected to Cognee ({mode}) at {service_url}")
    return client


async def _serve_cloud(
    management_url: Optional[str] = None,
    auth0_domain: Optional[str] = None,
    auth0_client_id: Optional[str] = None,
    auth0_audience: Optional[str] = None,
) -> CloudClient:
    """Full cloud flow: Auth0 Device Code → tenant discovery → API key → connect."""
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import (
        CloudCredentials,
        is_token_expired,
        load_credentials,
        save_credentials,
    )
    from cognee.api.v1.serve.device_auth import (
        device_code_login,
        extract_email_from_id_token,
        refresh_access_token,
    )
    from cognee.api.v1.serve.management_api import (
        create_tenant,
        get_current_tenant,
        get_or_create_api_key,
        get_service_url,
    )
    from cognee.api.v1.serve.state import set_remote_client

    mgmt_url = management_url or os.getenv(
        "COGNEE_CLOUD_URL", "https://api.dev.cloud.topoteretes.com"
    )
    mgmt_url = mgmt_url.rstrip("/")

    # Step 1: Check for saved cloud credentials (profiles saved by direct
    # mode carry no Auth0 session and must not shadow the cloud one).
    creds = load_credentials(cloud=True)

    if creds and creds.service_url and creds.api_key:
        client = CloudClient(creds.service_url, creds.api_key)
        try:
            if await client._health_check():
                logger.info("Using cached credentials for %s (token status ignored)", creds.email)
                set_remote_client(client)
                print(f"  Connected to Cognee Cloud at {creds.service_url}")
                return client
        except Exception as e:
            logger.warning("Immediate health check failed: %s", e)
        await client.close()

        if not is_token_expired(creds):
            logger.warning("Saved service URL unreachable, re-authenticating")
        elif creds.refresh_token:
            try:
                logger.info("Refreshing expired token for %s", creds.email)
                token = await refresh_access_token(
                    creds.refresh_token,
                    domain=auth0_domain,
                    client_id=auth0_client_id,
                )
                creds.access_token = token.access_token
                if token.refresh_token:
                    creds.refresh_token = token.refresh_token
                creds.expires_at = time.time() + token.expires_in
                save_credentials(creds)

                client = CloudClient(creds.service_url, creds.api_key)
                if await client._health_check():
                    set_remote_client(client)
                    print(f"  Connected to Cognee Cloud at {creds.service_url}")
                    return client
                await client.close()
            except Exception as e:
                logger.warning("Token refresh failed, re-authenticating: %s", e)

    # Step 2: Device Code Flow
    print("  Authenticating with Cognee Cloud...")
    token = await device_code_login(
        domain=auth0_domain,
        client_id=auth0_client_id,
        audience=auth0_audience,
    )

    access_token = token.access_token
    email = extract_email_from_id_token(token.id_token) if token.id_token else None

    # Step 3: Discover or create tenant
    tenant = await get_current_tenant(mgmt_url, access_token)
    if not tenant:
        if not email:
            raise RuntimeError(
                "Could not extract email from token. "
                "Ensure the Auth0 app includes 'email' in the scope."
            )
        tenant = await create_tenant(mgmt_url, access_token, email)

    # Step 4: Get service URL
    service_url = await get_service_url(mgmt_url, access_token)

    # Step 5: Get or create API key
    api_key = await get_or_create_api_key(mgmt_url, access_token)

    # Step 6: Save credentials
    creds = CloudCredentials(
        access_token=access_token,
        refresh_token=token.refresh_token,
        expires_at=time.time() + token.expires_in,
        service_url=service_url,
        api_key=api_key,
        management_url=mgmt_url,
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        email=email or "",
    )
    save_credentials(creds)

    # Step 7: Connect
    client = CloudClient(service_url, api_key)

    health_ok = await client._health_check()
    if not health_ok:
        logger.warning(
            "Service URL %s not responding to health check — may still be starting",
            service_url,
        )

    set_remote_client(client)
    print(f"  Connected to Cognee Cloud at {service_url}")
    if email:
        print(f"  Tenant: {tenant.name} ({email})")

    return client
