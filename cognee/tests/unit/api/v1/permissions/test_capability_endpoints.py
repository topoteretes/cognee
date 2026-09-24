"""The capability endpoints, exercised through the router.

The methods behind these routes are unit tested elsewhere. What is only
testable here is the wiring: which collaborator each handler calls, with which
arguments, and what the caller ends up seeing. That gap is not theoretical --
an earlier revision's role routes called get_role(role_id) against a get_role
that takes (tenant_id, role_name), so they raised TypeError before reaching
their permission check, and no method-level test could see it.

Collaborators are stubbed with real functions rather than bare mocks so a call
with the wrong arity fails here the way it would in production.
"""

import importlib
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.exc import NoResultFound

from cognee.api.v1.permissions.routers import get_permissions_router
from cognee.exceptions import CogneeApiError
from cognee.modules.users.exceptions import (
    CapabilityDeniedError,
    PermissionDeniedError,
    TenantNotFoundError,
)
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions.permission_types import (
    GRANT_CAPABILITIES,
    REVOKE_CAPABILITIES,
)

USER = SimpleNamespace(id=uuid4(), email="caller@example.com", tenant_id=uuid4())

_PREFIX = "/api/v1/permissions"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_permissions_router(), prefix=_PREFIX)

    async def override_user():
        return USER

    app.dependency_overrides[get_authenticated_user] = override_user

    # The real app registers this in cognee/api/client.py. Without it the
    # cognee exceptions these routes raise surface as 500 instead of their own
    # status code, which is precisely what the 403 cases assert.
    @app.exception_handler(CogneeApiError)
    async def cognee_error_handler(_, exc: CogneeApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": f"{exc.message} [{exc.name}]"},
        )

    return TestClient(app, raise_server_exceptions=False)


# Modules behind the capability routes that look collaborators up. Loaded with
# import_module because the methods package re-exports functions under the same
# names as these modules, so attribute access on the package returns the
# function, not the module.
_CAPABILITY_METHOD_MODULES = [
    importlib.import_module(f"cognee.modules.users.capabilities.methods.{module_name}")
    for module_name in (
        "authorized_get_effective_capabilities",
        "authorized_grant_capability",
        "authorized_revoke_capability",
        "get_capability_scope",
        "get_unheld_capabilities",
    )
]


@contextmanager
def _stub(name, impl):
    """Patch a collaborator in every capability method module that uses it.

    Each module imports its collaborators when it loads, so the name has to be
    replaced where it is looked up, not on the package it came from.
    """
    with ExitStack() as stack:
        patched = [
            stack.enter_context(patch.object(module, name, impl))
            for module in _CAPABILITY_METHOD_MODULES
            if hasattr(module, name)
        ]
        assert patched, f"no capability method module looks up {name}"
        yield


async def _allow_membership(user_id, tenant_id):
    return True


async def _allow_capability(requester_id, tenant_id, grant_type):
    return True


async def _deny_capability(requester_id, tenant_id, grant_type):
    raise CapabilityDeniedError(grant_type)


def _principal(kind, tenant_id=None):
    """get_principal loads subclass columns, so a role arrives with its tenant_id."""

    async def get_principal(principal_id):
        return SimpleNamespace(id=principal_id, type=kind, tenant_id=tenant_id)

    return get_principal


async def _tenant_exists(tenant_id):
    return SimpleNamespace(id=tenant_id)


class TestReadingOwnCapabilities:
    def test_returns_the_sorted_capability_names(self, client):
        tenant_id = uuid4()

        async def capabilities(user_id, requested_tenant_id):
            assert user_id == USER.id
            assert str(requested_tenant_id) == str(tenant_id)
            return {"manage_users", "aaa_sorts_first"}

        with (
            _stub("require_tenant_membership", _allow_membership),
            _stub("get_effective_capabilities", capabilities),
        ):
            response = client.get(f"{_PREFIX}/tenants/{tenant_id}/capabilities/me")

        assert response.status_code == 200
        assert response.json() == {"capabilities": ["aaa_sorts_first", "manage_users"]}

    def test_a_foreign_tenant_and_an_unknown_one_are_indistinguishable(self, client):
        """The point of routing both through require_tenant_membership.

        If a real tenant answered differently from a made-up id, any
        authenticated user could enumerate tenant ids.
        """

        async def deny(user_id, tenant_id):
            raise PermissionDeniedError(message="User is not a member of this tenant")

        async def capabilities(user_id, tenant_id):
            raise AssertionError("resolution must not run for a non-member")

        with (
            _stub("require_tenant_membership", deny),
            _stub("get_effective_capabilities", capabilities),
        ):
            foreign = client.get(f"{_PREFIX}/tenants/{uuid4()}/capabilities/me")
            unknown = client.get(f"{_PREFIX}/tenants/{uuid4()}/capabilities/me")

        assert foreign.status_code == 403
        assert unknown.status_code == 403
        assert foreign.json() == unknown.json()


class TestGrantScope:
    """One endpoint, three principal kinds: the scope has to follow the kind."""

    def test_a_tenant_principal_is_its_own_scope(self, client):
        tenant_id = uuid4()
        granted = []

        async def grant(principal_id, scope_tenant_id, capabilities, granted_by=None):
            granted.append((str(principal_id), str(scope_tenant_id), capabilities))

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_grant_permission", _allow_capability),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{tenant_id}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert granted == [(str(tenant_id), str(tenant_id), ["manage_users"])]

    def test_a_role_principal_is_scoped_and_authorized_against_its_owning_tenant(self, client):
        role_id = uuid4()
        owning_tenant_id = uuid4()
        authorized_against = []
        granted = []

        async def has_capability(requester_id, tenant_id, grant_type):
            authorized_against.append((str(tenant_id), grant_type))
            return True

        async def grant(principal_id, scope_tenant_id, capabilities, granted_by=None):
            granted.append((str(principal_id), str(scope_tenant_id), capabilities))

        with (
            _stub("get_principal", _principal("role", tenant_id=owning_tenant_id)),
            _stub("get_tenant", _tenant_exists),
            _stub("has_grant_permission", has_capability),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{role_id}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        # Not the caller's own tenant: authorization follows the role's owner,
        # so a caller cannot grant into a tenant they do not administer. The
        # second check is that they hold what they grant, in that same tenant.
        assert authorized_against == [
            (str(owning_tenant_id), GRANT_CAPABILITIES),
            (str(owning_tenant_id), "manage_users"),
        ]
        assert granted == [(str(role_id), str(owning_tenant_id), ["manage_users"])]

    def test_a_user_principal_without_a_tenant_defaults_to_the_callers_current_one(self, client):
        """Never the target's users.tenant_id: a person can belong to several
        tenants. The caller's current tenant is the one they act in, as for
        create_role."""
        person_id = uuid4()
        granted = []

        async def get_user(user_id):
            assert user_id == USER.id
            return SimpleNamespace(id=user_id, tenant_id=USER.tenant_id)

        async def grant(principal_id, scope_tenant_id, capabilities, granted_by=None):
            granted.append((str(principal_id), str(scope_tenant_id), capabilities))

        with (
            _stub("get_principal", _principal("user")),
            _stub("get_user", get_user),
            _stub("get_tenant", _tenant_exists),
            _stub("has_grant_permission", _allow_capability),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{person_id}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert granted == [(str(person_id), str(USER.tenant_id), ["manage_users"])]

    def test_a_user_principal_with_a_tenant_is_granted_in_that_tenant(self, client):
        person_id = uuid4()
        tenant_id = uuid4()
        granted = []

        async def grant(principal_id, scope_tenant_id, capabilities, granted_by=None):
            granted.append((str(principal_id), str(scope_tenant_id), capabilities))

        with (
            _stub("get_principal", _principal("user")),
            _stub("get_tenant", _tenant_exists),
            _stub("has_grant_permission", _allow_capability),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{person_id}",
                params={"capability": "manage_users", "tenant_id": str(tenant_id)},
            )

        assert response.status_code == 200
        assert granted == [(str(person_id), str(tenant_id), ["manage_users"])]

    @pytest.mark.parametrize("method", ["POST", "DELETE"])
    def test_an_unknown_principal_reads_as_missing_permission(self, client, method):
        """A principal that does not exist must not be distinguishable from one
        the caller may not touch, or the endpoint enumerates real ids.

        Grant and revoke are gated by different capabilities, so each has to
        match its own denial, not a shared one.
        """

        async def missing(principal_id):
            raise NoResultFound("No row was found when one was required")

        with (
            _stub("get_principal", missing),
            _stub("has_grant_permission", _allow_capability),
        ):
            unknown = client.request(
                method,
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_grant_permission", _deny_capability),
        ):
            forbidden = client.request(
                method,
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        assert unknown.status_code == 403
        assert forbidden.status_code == 403
        assert unknown.json() == forbidden.json()

    @pytest.mark.parametrize("method", ["POST", "DELETE"])
    def test_a_made_up_tenant_reads_as_a_foreign_one(self, client, method):
        """Resolution raises TenantNotFoundError (404) for a tenant id that does
        not exist; left alone, anyone could tell real tenant ids from made-up
        ones by comparing that with the 403 a foreign tenant gets."""

        async def no_such_tenant(tenant_id):
            raise TenantNotFoundError(message=f"Could not find tenant: {tenant_id}")

        with (
            _stub("get_principal", _principal("user")),
            _stub("get_tenant", no_such_tenant),
            _stub("has_grant_permission", _allow_capability),
        ):
            made_up = client.request(
                method,
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users", "tenant_id": str(uuid4())},
            )

        with (
            _stub("get_principal", _principal("user")),
            _stub("get_tenant", _tenant_exists),
            _stub("has_grant_permission", _deny_capability),
        ):
            foreign = client.request(
                method,
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users", "tenant_id": str(uuid4())},
            )

        assert made_up.status_code == 403
        assert foreign.status_code == 403
        assert made_up.json() == foreign.json()


class TestValidationAndAuthorization:
    @pytest.mark.parametrize("capability", ["read", "write", "delete", "share", "not_a_capability"])
    def test_names_outside_the_catalog_are_rejected(self, client, capability):
        """Dataset permissions are ACL business, and an arbitrary name gates
        nothing; storing either would look like it worked while doing nothing."""

        async def get_principal(principal_id):
            raise AssertionError("validation must run before any lookup")

        with (
            _stub("get_principal", get_principal),
            _stub("has_grant_permission", _allow_capability),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": capability},
            )

        assert response.status_code == 400
        assert "Unknown capability" in response.json()["detail"]

    def test_a_caller_without_grant_capabilities_is_refused(self, client):
        checked = []

        async def deny(requester_id, tenant_id, grant_type):
            checked.append(grant_type)
            raise CapabilityDeniedError(grant_type)

        async def grant(principal_id, scope_tenant_id, capabilities, granted_by=None):
            raise AssertionError("must not reach the write")

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_grant_permission", deny),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 403
        assert checked == [GRANT_CAPABILITIES]
        assert "grant capabilities" in response.json()["detail"]


class TestRevoking:
    def test_revoke_reaches_the_method_with_the_same_scope_rules(self, client):
        role_id = uuid4()
        owning_tenant_id = uuid4()
        authorized_against = []
        revoked = []

        async def has_capability(requester_id, tenant_id, grant_type):
            authorized_against.append((str(tenant_id), grant_type))
            return True

        async def revoke(principal_id, scope_tenant_id, capabilities):
            revoked.append((str(principal_id), str(scope_tenant_id), capabilities))

        with (
            _stub("get_principal", _principal("role", tenant_id=owning_tenant_id)),
            _stub("get_tenant", _tenant_exists),
            _stub("has_grant_permission", has_capability),
            _stub("revoke_capability", revoke),
        ):
            response = client.request(
                "DELETE",
                f"{_PREFIX}/capabilities/{role_id}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert authorized_against == [(str(owning_tenant_id), REVOKE_CAPABILITIES)]
        assert revoked == [(str(role_id), str(owning_tenant_id), ["manage_users"])]

    def test_a_caller_without_revoke_capabilities_is_refused(self, client):
        """Holding GRANT_CAPABILITIES is not enough: revoking is its own check."""
        checked = []

        async def deny(requester_id, tenant_id, grant_type):
            checked.append(grant_type)
            raise CapabilityDeniedError(grant_type)

        async def revoke(principal_id, scope_tenant_id, capabilities):
            raise AssertionError("must not reach the write")

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_grant_permission", deny),
            _stub("revoke_capability", revoke),
        ):
            response = client.request(
                "DELETE",
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 403
        assert checked == [REVOKE_CAPABILITIES]
        assert "revoke capabilities" in response.json()["detail"]


class TestBatches:
    """Several capabilities in one request, for a frontend that edits a whole
    set of checkboxes at once."""

    def test_repeated_parameters_grant_them_together_and_record_the_granter(self, client):
        tenant_id = uuid4()
        writes = []

        async def grant(principal_id, scope_tenant_id, capabilities, granted_by=None):
            writes.append((capabilities, str(granted_by)))

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_grant_permission", _allow_capability),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{tenant_id}",
                params=[("capability", "manage_users"), ("capability", "revoke_capabilities")],
            )

        assert response.status_code == 200
        # One write for the whole batch, so it lands in one transaction.
        assert writes == [(["manage_users", "revoke_capabilities"], str(USER.id))]

    def test_repeated_parameters_revoke_them_together(self, client):
        tenant_id = uuid4()
        writes = []

        async def revoke(principal_id, scope_tenant_id, capabilities):
            writes.append(capabilities)

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_grant_permission", _allow_capability),
            _stub("revoke_capability", revoke),
        ):
            response = client.request(
                "DELETE",
                f"{_PREFIX}/capabilities/{tenant_id}",
                params=[("capability", "manage_users"), ("capability", "grant_capabilities")],
            )

        assert response.status_code == 200
        assert writes == [["manage_users", "grant_capabilities"]]

    @pytest.mark.parametrize("method", ["POST", "DELETE"])
    def test_one_unknown_name_rejects_the_whole_batch(self, client, method):
        """Half a batch applied is worse than none: the caller cannot tell
        which part landed without reading everything back."""

        async def write(*args, **kwargs):
            raise AssertionError("nothing may be written when any name is unknown")

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_grant_permission", _allow_capability),
            _stub("grant_capability", write),
            _stub("revoke_capability", write),
        ):
            response = client.request(
                method,
                f"{_PREFIX}/capabilities/{uuid4()}",
                params=[
                    ("capability", "manage_users"),
                    ("capability", "read"),
                    ("capability", "not_a_capability"),
                ],
            )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "read" in detail and "not_a_capability" in detail
