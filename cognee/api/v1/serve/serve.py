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
            discovery. Can also be set via ``COGNEE_SERVICE_URL`` env var.
        api_key: API key for authentication. Used with ``url`` for direct
            connections, or via ``COGNEE_API_KEY`` env var.
        management_url: Override the Management API URL (cloud mode only).
        auth0_domain: Override the Auth0 domain (cloud mode only).
        auth0_client_id: Override the Auth0 Device Code client ID.
        auth0_audience: Override the Auth0 API audience.

    Returns:
        CloudClient connected to the instance.
    """
    # Resolve URL from arg or env
    service_url = url or os.getenv("COGNEE_SERVICE_URL")
    resolved_api_key = api_key or os.getenv("COGNEE_API_KEY", "")

    if service_url:
        return await _serve_direct(service_url, resolved_api_key)

    return await _serve_cloud(
        management_url=management_url,
        auth0_domain=auth0_domain,
        auth0_client_id=auth0_client_id,
        auth0_audience=auth0_audience,
    )


async def _serve_direct(service_url: str, api_key: str = "") -> CloudClient:
    """Connect directly to a Cognee instance — no Auth0, no Management API."""
    from cognee.api.v1.serve.cloud_client import CloudClient
    from cognee.api.v1.serve.credentials import CloudCredentials, save_credentials
    from cognee.api.v1.serve.state import set_remote_client

    service_url = service_url.rstrip("/")
    client = CloudClient(service_url, api_key)

    health_ok = await client._health_check()
    if not health_ok:
        logger.warning("Instance at %s did not respond to health check", service_url)

    # Save so subsequent serve() calls reconnect without args
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

    # Step 1: Check for saved credentials
    creds = load_credentials()

    if creds and creds.service_url and creds.api_key:
        if not is_token_expired(creds):
            logger.info("Using saved credentials for %s", creds.email)
            client = CloudClient(creds.service_url, creds.api_key)
            if await client._health_check():
                set_remote_client(client)
                print(f"  Connected to Cognee Cloud at {creds.service_url}")
                return client
            else:
                logger.warning("Saved service URL unreachable, re-authenticating")
                await client.close()

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
