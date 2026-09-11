"""The social layer of a COGX archive is a privilege boundary.

`permissions.json` supplies emails, password hashes and account flags verbatim,
and every path that consumes it mints accounts through `_ensure_user`, which
writes `hashed_password` and `is_superuser` straight from the payload. So the
superuser check has to gate the LAYER, not one path through it.

Regression: the check sat behind `owner_payload is None`, so an archive whose
permissions.json carried `grants` but no `owner` returned before the check ever
ran -- and `_apply_social_grants`, reached independently from `import_source`,
then created accounts from attacker-controlled payloads. Any caller who could
import an archive could mint a superuser with a password hash of their choosing.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.migration.import_source import (
    _apply_social_grants,
    _resolve_import_user,
)
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError

# A layer with grants and deliberately NO "owner" key -- the shape that used to
# slip past the gate.
_GRANTS_ONLY = {
    "grants": [
        {
            "user": {
                "email": "attacker@example.com",
                "hashed_password": "$2b$12$attackerchosenhashvalue",
                "is_superuser": True,
                "is_active": True,
                "is_verified": True,
            },
            "permissions": ["read", "write"],
        }
    ]
}

_WITH_OWNER = {"owner": {"email": "owner@example.com", "hashed_password": "x"}, **_GRANTS_ONLY}


def _user(is_superuser: bool):
    return SimpleNamespace(id=uuid4(), is_superuser=is_superuser, email="importer@example.com")


@pytest.mark.parametrize("layer", [_GRANTS_ONLY, _WITH_OWNER], ids=["grants-only", "with-owner"])
@pytest.mark.asyncio
async def test_non_superuser_cannot_import_any_social_layer(layer):
    with pytest.raises(PermissionDeniedError):
        await _resolve_import_user(SimpleNamespace(social_layer=layer), _user(is_superuser=False))


@pytest.mark.asyncio
async def test_grant_replay_refuses_a_non_superuser_importer():
    """The second entry point. `import_source` calls this directly on any truthy
    social layer, so it cannot rely on the caller having passed the gate."""
    importer = _user(is_superuser=False)
    with pytest.raises(PermissionDeniedError):
        await _apply_social_grants(
            SimpleNamespace(social_layer=_GRANTS_ONLY),
            "some-dataset",
            owner=importer,
            importer=importer,
        )


@pytest.mark.asyncio
async def test_archives_without_a_social_layer_are_unaffected():
    """The common case must not start demanding a superuser."""
    caller = _user(is_superuser=False)
    assert await _resolve_import_user(SimpleNamespace(social_layer=None), caller) is caller
    assert await _resolve_import_user(SimpleNamespace(social_layer={}), caller) is caller
    assert await _resolve_import_user(SimpleNamespace(), caller) is caller
