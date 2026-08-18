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

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.api.v1.permissions.routers import get_permissions_router
from cognee.exceptions import CogneeApiError
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.permissions import methods as permission_methods

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


def _stub(name, impl):
    """Patch a collaborator on the methods package.

    The handlers import from that package at call time, so this is the name
    they resolve.
    """
    return patch.object(permission_methods, name, impl)


async def _allow_membership(user_id, tenant_id):
    return True


async def _allow_management(requester_id, tenant_id):
    return True


def _principal(kind):
    async def get_principal(principal_id):
        return SimpleNamespace(id=principal_id, type=kind)

    return get_principal


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

        async def grant(principal_id, scope_tenant_id, capability):
            granted.append((str(principal_id), str(scope_tenant_id), capability))

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_user_management_permission", _allow_management),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{tenant_id}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert granted == [(str(tenant_id), str(tenant_id), "manage_users")]

    def test_a_role_principal_is_scoped_and_authorized_against_its_owning_tenant(self, client):
        role_id = uuid4()
        owning_tenant_id = uuid4()
        authorized_against = []
        granted = []

        async def get_role_by_id(requested_role_id):
            assert str(requested_role_id) == str(role_id)
            return SimpleNamespace(id=role_id, tenant_id=owning_tenant_id)

        async def has_management(requester_id, tenant_id):
            authorized_against.append(str(tenant_id))
            return True

        async def grant(principal_id, scope_tenant_id, capability):
            granted.append((str(principal_id), str(scope_tenant_id), capability))

        with (
            _stub("get_principal", _principal("role")),
            _stub("get_role_by_id", get_role_by_id),
            _stub("has_user_management_permission", has_management),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{role_id}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        # Not the caller's own tenant: authorization follows the role's owner,
        # so a caller cannot grant into a tenant they do not administer.
        assert authorized_against == [str(owning_tenant_id)]
        assert granted == [(str(role_id), str(owning_tenant_id), "manage_users")]

    def test_a_user_principal_requires_an_explicit_tenant(self, client):
        """A person can belong to several tenants, so the scope cannot be derived."""

        async def grant(principal_id, scope_tenant_id, capability):
            raise AssertionError("must not write without a scope")

        with (
            _stub("get_principal", _principal("user")),
            _stub("has_user_management_permission", _allow_management),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 400
        assert "tenant_id is required" in response.json()["detail"]

    def test_a_user_principal_with_a_tenant_is_granted_in_that_tenant(self, client):
        person_id = uuid4()
        tenant_id = uuid4()
        granted = []

        async def grant(principal_id, scope_tenant_id, capability):
            granted.append((str(principal_id), str(scope_tenant_id), capability))

        with (
            _stub("get_principal", _principal("user")),
            _stub("has_user_management_permission", _allow_management),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{person_id}",
                params={"capability": "manage_users", "tenant_id": str(tenant_id)},
            )

        assert response.status_code == 200
        assert granted == [(str(person_id), str(tenant_id), "manage_users")]

    def test_an_unknown_principal_reads_as_missing_permission(self, client):
        """A principal that does not exist must not be distinguishable from one
        the caller may not touch, or the endpoint enumerates real ids."""

        async def missing(principal_id):
            raise LookupError("no such principal")

        async def deny(requester_id, tenant_id):
            raise PermissionDeniedError(
                message="User is not authorized to manage users for this tenant"
            )

        with (
            _stub("get_principal", missing),
            _stub("has_user_management_permission", _allow_management),
        ):
            unknown = client.post(
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_user_management_permission", deny),
        ):
            forbidden = client.post(
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        assert unknown.status_code == 403
        assert forbidden.status_code == 403
        assert unknown.json() == forbidden.json()


class TestValidationAndAuthorization:
    @pytest.mark.parametrize("capability", ["read", "write", "delete", "share", "not_a_capability"])
    def test_names_outside_the_catalog_are_rejected(self, client, capability):
        """Dataset permissions are ACL business, and an arbitrary name gates
        nothing; storing either would look like it worked while doing nothing."""

        async def get_principal(principal_id):
            raise AssertionError("validation must run before any lookup")

        with (
            _stub("get_principal", get_principal),
            _stub("has_user_management_permission", _allow_management),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": capability},
            )

        assert response.status_code == 400
        assert "Unknown capability" in response.json()["detail"]

    def test_a_caller_without_user_management_is_refused(self, client):
        async def deny(requester_id, tenant_id):
            raise PermissionDeniedError(
                message="User is not authorized to manage users for this tenant"
            )

        async def grant(principal_id, scope_tenant_id, capability):
            raise AssertionError("must not reach the write")

        with (
            _stub("get_principal", _principal("tenant")),
            _stub("has_user_management_permission", deny),
            _stub("grant_capability", grant),
        ):
            response = client.post(
                f"{_PREFIX}/capabilities/{uuid4()}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 403


class TestRevoking:
    def test_revoke_reaches_the_method_with_the_same_scope_rules(self, client):
        role_id = uuid4()
        owning_tenant_id = uuid4()
        revoked = []

        async def get_role_by_id(requested_role_id):
            return SimpleNamespace(id=role_id, tenant_id=owning_tenant_id)

        async def revoke(principal_id, scope_tenant_id, capability):
            revoked.append((str(principal_id), str(scope_tenant_id), capability))

        with (
            _stub("get_principal", _principal("role")),
            _stub("get_role_by_id", get_role_by_id),
            _stub("has_user_management_permission", _allow_management),
            _stub("revoke_capability", revoke),
        ):
            response = client.request(
                "DELETE",
                f"{_PREFIX}/capabilities/{role_id}",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert revoked == [(str(role_id), str(owning_tenant_id), "manage_users")]
