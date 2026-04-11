"""Management API client — tenant discovery and API key provisioning.

Replicates the frontend's TenantProvider.tsx flow in Python.
"""

import asyncio
import hashlib
import os
from dataclasses import dataclass
from typing import Optional

import aiohttp

from cognee.shared.logging_utils import get_logger

logger = get_logger("serve.management_api")

DEFAULT_MANAGEMENT_URL = "https://api.dev.cloud.topoteretes.com"


def _get_management_url() -> str:
    return os.getenv("COGNEE_CLOUD_URL", DEFAULT_MANAGEMENT_URL).rstrip("/")


@dataclass
class Tenant:
    id: str
    name: str


def _email_to_tenant_name(email: str) -> str:
    """Generate a deterministic tenant name from email, matching the frontend convention."""
    # Frontend uses uuid5(NAMESPACE_URL, email) — we replicate the same hash
    from uuid import uuid5, NAMESPACE_URL

    return f"tenant-{uuid5(NAMESPACE_URL, email)}"


async def get_current_tenant(
    management_url: str,
    access_token: str,
) -> Optional[Tenant]:
    """GET /api/tenants/current — returns the user's active tenant or None."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{management_url}/api/tenants/current",
            headers={"Authorization": f"Bearer {access_token}"},
        ) as resp:
            if resp.status == 404:
                return None
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Failed to get tenant ({resp.status}): {body}")
            data = await resp.json()
            return Tenant(id=str(data.get("id", "")), name=data.get("name", ""))


async def create_tenant(
    management_url: str,
    access_token: str,
    email: str,
    poll_timeout: int = 300,
    poll_interval: int = 5,
) -> Tenant:
    """Create a tenant and poll until it's ready.

    Follows the frontend TenantProvider pattern: create, then poll
    get_current_tenant until the tenant appears (provisioning can take
    up to a few minutes).
    """
    tenant_name = _email_to_tenant_name(email)
    logger.info("Creating tenant '%s' for %s", tenant_name, email)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{management_url}/api/tenants",
            params={"tenant_name": tenant_name},
            headers={"Authorization": f"Bearer {access_token}"},
        ) as resp:
            if resp.status not in (200, 201, 202):
                body = await resp.text()
                raise RuntimeError(f"Failed to create tenant ({resp.status}): {body}")

    # Poll until tenant is available
    print("  Provisioning tenant (this may take a minute)...")
    elapsed = 0
    while elapsed < poll_timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        tenant = await get_current_tenant(management_url, access_token)
        if tenant and tenant.id:
            logger.info("Tenant ready: %s (%s)", tenant.name, tenant.id)
            return tenant

    raise TimeoutError(f"Tenant provisioning timed out after {poll_timeout}s")


async def get_service_url(
    management_url: str,
    access_token: str,
) -> str:
    """GET /api/tenants/current/service-url — returns the tenant's dedicated instance URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{management_url}/api/tenants/current/service-url",
            headers={"Authorization": f"Bearer {access_token}"},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Failed to get service URL ({resp.status}): {body}")
            data = await resp.json()
            url = data.get("service_url") or data.get("url", "")
            if not url:
                raise RuntimeError("Service URL is empty — tenant may still be provisioning")
            return url.rstrip("/")


async def get_or_create_api_key(
    management_url: str,
    access_token: str,
    max_retries: int = 3,
) -> str:
    """Get an existing API key or create one for the tenant instance."""
    headers = {"Authorization": f"Bearer {access_token}"}

    async with aiohttp.ClientSession() as session:
        # Try to get existing keys
        async with session.get(
            f"{management_url}/api/api-keys",
            headers=headers,
        ) as resp:
            if resp.status == 200:
                keys = await resp.json()
                if isinstance(keys, list) and keys:
                    # Return the first usable key
                    key = keys[0].get("key") or keys[0].get("api_key", "")
                    if key:
                        return key

        # Create a new key with retries
        for attempt in range(max_retries):
            async with session.post(
                f"{management_url}/api/api-keys",
                headers=headers,
            ) as resp:
                if resp.status in (200, 201):
                    data = await resp.json()
                    key = data.get("key") or data.get("api_key", "")
                    if key:
                        return key
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise RuntimeError("Failed to create API key after retries")
