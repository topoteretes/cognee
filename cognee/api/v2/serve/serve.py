"""Top-level serve() orchestrator — connects the SDK to Cognee Cloud."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cognee.api.v2.serve.cloud_client import CloudClient

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve")


async def serve(
    management_url: Optional[str] = None,
    auth0_domain: Optional[str] = None,
    auth0_client_id: Optional[str] = None,
    auth0_audience: Optional[str] = None,
) -> "CloudClient":
    """Connect the local Cognee SDK to a remote Cognee Cloud instance.

    Authenticates via the Auth0 Device Code Flow (opens a browser URL),
    discovers or creates a tenant, obtains an API key, and sets all V2
    operations (remember, recall, improve, forget) to route to the cloud.

    Credentials are cached at ``~/.cognee/cloud_credentials.json`` so
    subsequent calls skip the device flow when the token is still valid.

    Args:
        management_url: Override the Management API URL. Defaults to
            ``COGNEE_CLOUD_URL`` env var or ``https://api.dev.cloud.topoteretes.com``.
        auth0_domain: Override the Auth0 domain.
        auth0_client_id: Override the Auth0 Device Code client ID.
        auth0_audience: Override the Auth0 API audience.

    Returns:
        CloudClient connected to the remote instance.

    Example::

        import cognee

        # Authenticate and connect
        await cognee.serve()

        # All V2 ops now route to the cloud
        await cognee.remember("Einstein was born in Ulm.")
        results = await cognee.recall("Where was Einstein born?")

        # Disconnect to go back to local mode
        await cognee.disconnect()
    """
    from cognee.api.v2.serve.cloud_client import CloudClient
    from cognee.api.v2.serve.credentials import (
        CloudCredentials,
        is_token_expired,
        load_credentials,
        save_credentials,
    )
    from cognee.api.v2.serve.device_auth import (
        device_code_login,
        extract_email_from_id_token,
        refresh_access_token,
    )
    from cognee.api.v2.serve.management_api import (
        create_tenant,
        get_current_tenant,
        get_or_create_api_key,
        get_service_url,
    )
    from cognee.api.v2.serve.state import set_remote_client

    mgmt_url = management_url or os.getenv(
        "COGNEE_CLOUD_URL", "https://api.dev.cloud.topoteretes.com"
    )
    mgmt_url = mgmt_url.rstrip("/")

    # Step 1: Check for saved credentials
    creds = load_credentials()

    if creds and creds.service_url and creds.api_key:
        if not is_token_expired(creds):
            # Credentials still valid — connect directly
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
            # Token expired but we have a refresh token
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
            "Service URL %s not responding to health check — may still be starting", service_url
        )

    set_remote_client(client)
    print(f"  Connected to Cognee Cloud at {service_url}")
    if email:
        print(f"  Tenant: {tenant.name} ({email})")

    return client
