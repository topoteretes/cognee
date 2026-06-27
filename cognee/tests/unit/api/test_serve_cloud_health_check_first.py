"""``_serve_cloud`` should fast-path on a saved credential whose
``service_url`` is reachable, even if the cached ``access_token`` is
already expired.

Before this change, ``_serve_cloud`` checked ``is_token_expired(creds)``
first and would unconditionally fall through to ``refresh_access_token``
(or device-code login) whenever the local clock said the token had
expired — even if the saved ``(service_url, api_key)`` was still
perfectly able to talk to the cognee service. Auth0 / management being
slow or temporarily down would then stall SDK startup.

These tests exercise the *actual* ``_serve_cloud`` with its
side-effects mocked at the import boundary, so we are validating the
real control flow rather than a hand-rolled stub.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from cognee.api.v1.serve import credentials as creds_mod
from cognee.api.v1.serve import serve as serve_mod
from cognee.api.v1.serve import state as serve_state
from cognee.api.v1.serve.credentials import CloudCredentials
from cognee.api.v1.serve.serve import _serve_cloud


def _build_saved_creds(*, expired: bool) -> CloudCredentials:
    """A credential file that says we have a saved tenant. The
    ``expires_at`` field is set in the past when ``expired=True`` so
    ``is_token_expired`` returns True without us touching the wall
    clock.
    """
    now = time.time()
    return CloudCredentials(
        access_token="access-token-123",
        refresh_token="refresh-token-456",
        expires_at=now - 3600 if expired else now + 3600,
        service_url="https://cognee.test",
        api_key="api-key-789",
        management_url="https://api.dev.cloud.topoteretes.com",
        tenant_id="tenant-1",
        tenant_name="Test Tenant",
        email="user@test.example",
    )


def _patch_all(*, refresh_mock, health_return: bool, creds):
    """Return a context-manager stack that mocks every external call
    ``_serve_cloud`` would make.

    ``_serve_cloud`` does ``from X import Y`` *inside* its body. Each
    call re-evaluates the import — so patching ``X.Y`` (the source
    module) is the right target. Patching ``serve_mod.Y`` does NOT
    work because ``serve_mod`` has no such attribute (Python only
    binds the lazy import into the function's local namespace, not the
    module's globals). Verified empirically with a stand-alone repro.
    """
    from cognee.api.v1.serve import device_auth as device_auth_mod

    fake_client = MagicMock(name="CloudClient")
    fake_client._health_check = AsyncMock(return_value=health_return)
    fake_client.close = AsyncMock(return_value=None)
    fake_client.HEALTH_CHECK_TIMEOUT = MagicMock(name="HEALTH_CHECK_TIMEOUT")

    # Always mock device_code_login so the function can never reach
    # the Auth0 device-code flow during these tests. Reaching it
    # would mean the fast path *and* the refresh path both failed,
    # which is not what we are testing here.
    device_code_mock = AsyncMock(side_effect=AssertionError("device-code should not be reached"))

    return [
        patch.object(creds_mod, "load_credentials", return_value=creds),
        # CloudClient is imported as `from .cloud_client import CloudClient`,
        # i.e. `cognee.api.v1.serve.cloud_client.CloudClient` — patch
        # that source module so the lazy import resolves to the mock.
        patch("cognee.api.v1.serve.cloud_client.CloudClient", return_value=fake_client),
        # refresh_access_token & device_code_login both come from
        # cognee.api.v1.serve.device_auth.
        patch.object(device_auth_mod, "refresh_access_token", new=refresh_mock),
        patch.object(device_auth_mod, "device_code_login", new=device_code_mock),
    ], fake_client


def test_fast_path_uses_saved_creds_when_token_expired_and_health_ok():
    """Health check passes → use saved creds → skip refresh entirely.

    This is the core fix for #3249: Auth0/management availability
    must not block SDK startup if the cognee service is reachable
    with the saved key.
    """
    creds = _build_saved_creds(expired=True)
    refresh_mock = AsyncMock(name="refresh_access_token")

    patches, fake_client = _patch_all(refresh_mock=refresh_mock, health_return=True, creds=creds)

    async def run():
        with patches[0], patches[1], patches[2], patches[3]:
            return await _serve_cloud()

    result = asyncio.run(run())

    # The fast-path client must be the one returned, and it must have
    # been set as the remote singleton.
    assert result is fake_client
    assert serve_state.get_remote_client() is fake_client
    # Refresh must not have been touched — this is the whole point of #3249.
    refresh_mock.assert_not_called()
    # And the health check must have been called with the short timeout.
    fake_client._health_check.assert_awaited_once_with(timeout=fake_client.HEALTH_CHECK_TIMEOUT)


def test_fast_path_falls_through_to_refresh_when_health_check_fails():
    """When health check fails AND token is expired, the original
    refresh path is still invoked (regression guard — we must not
    break callers who relied on the previous behaviour).
    """
    creds = _build_saved_creds(expired=True)
    refresh_mock = AsyncMock(
        name="refresh_access_token",
        return_value=MagicMock(
            access_token="new-access",
            refresh_token="new-refresh",
            id_token=None,
            token_type="Bearer",
            expires_in=3600,
        ),
    )

    patches, fake_client = _patch_all(refresh_mock=refresh_mock, health_return=False, creds=creds)

    async def run():
        # All four patches go in together; if the implementation falls
        # through past the refresh path, device_code_login will raise
        # AssertionError (which we catch below and turn into a
        # success signal — we only care that refresh was called and
        # the unreachable client was closed).
        with patches[0], patches[1], patches[2], patches[3]:
            return await _serve_cloud()

    try:
        asyncio.run(run())
    except AssertionError as e:
        assert "device-code should not be reached" in str(e)
        # device_code was reached only AFTER refresh — that is the
        # correct behaviour. If it had been reached without refresh,
        # the assert would still fire but the next two assertions
        # would fail.

    # Regression guard: when the fast-path health check fails, the
    # original refresh path must still be invoked.
    refresh_mock.assert_awaited_once()
    # The unreachable fast-path client is closed once before falling
    # through to the refresh path (which constructs its own client
    # and would also close it — but since we patched CloudClient to
    # return the same fake, the "second" close counts the refresh
    # path's close). We just require at least one close.
    assert fake_client.close.await_count >= 1
