"""The capability endpoints, exercised through the router.

The methods behind these routes are unit tested elsewhere. What is only
testable here is the wiring: which collaborator each handler calls, with which
arguments, and what the caller ends up seeing. That gap is not theoretical --
both role routes called get_role(role_id) against a get_role that takes
(tenant_id, role_name), so they raised TypeError before reaching their
permission check, and no method-level test could see it.

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
from cognee.modules.users.exceptions import PermissionDeniedError, RoleNotFoundError
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
    # status code, which is precisely what the 403/404 cases assert.
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


class TestTenantCapabilities:
    def test_grant_reaches_the_method_with_the_capability(self, client):
        tenant_id = uuid4()
        granted = []

        async def give(target_tenant_id, permission_name):
            granted.append((str(target_tenant_id), permission_name))

        with (
            _stub("has_user_management_permission", _allow_management),
            _stub("give_default_permission_to_tenant", give),
        ):
            response = client.post(
                f"{_PREFIX}/tenants/{tenant_id}/capabilities",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert granted == [(str(tenant_id), "manage_users")]

    def test_revoke_reaches_the_method_with_the_capability(self, client):
        tenant_id = uuid4()
        revoked = []

        async def revoke(target_tenant_id, permission_name):
            revoked.append((str(target_tenant_id), permission_name))

        with (
            _stub("has_user_management_permission", _allow_management),
            _stub("revoke_default_permission_from_tenant", revoke),
        ):
            response = client.request(
                "DELETE",
                f"{_PREFIX}/tenants/{tenant_id}/capabilities",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert revoked == [(str(tenant_id), "manage_users")]

    @pytest.mark.parametrize("capability", ["read", "write", "delete", "share", "not_a_capability"])
    def test_names_outside_the_catalog_are_rejected(self, client, capability):
        """Dataset permissions share the table, so granting one would write a
        row that capability resolution filters out again -- silently doing
        nothing while looking like it worked."""

        async def give(tenant_id, permission_name):
            raise AssertionError(f"{permission_name} must not be written")

        with (
            _stub("has_user_management_permission", _allow_management),
            _stub("give_default_permission_to_tenant", give),
        ):
            response = client.post(
                f"{_PREFIX}/tenants/{uuid4()}/capabilities",
                params={"capability": capability},
            )

        assert response.status_code == 400
        assert "Unknown capability" in response.json()["detail"]

    def test_a_caller_without_user_management_is_refused(self, client):
        async def deny(requester_id, tenant_id):
            raise PermissionDeniedError(
                message="User is not authorized to manage users for this tenant"
            )

        async def give(tenant_id, permission_name):
            raise AssertionError("must not reach the write")

        with (
            _stub("has_user_management_permission", deny),
            _stub("give_default_permission_to_tenant", give),
        ):
            response = client.post(
                f"{_PREFIX}/tenants/{uuid4()}/capabilities",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 403


class TestRoleCapabilities:
    """The branch that never ran: the lookup takes a role id alone."""

    def test_grant_looks_the_role_up_by_id_and_authorizes_against_its_tenant(self, client):
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

        async def give(target_role_id, permission_name):
            granted.append((str(target_role_id), permission_name))

        with (
            _stub("get_role_by_id", get_role_by_id),
            _stub("has_user_management_permission", has_management),
            _stub("give_default_permission_to_role", give),
        ):
            response = client.post(
                f"{_PREFIX}/roles/{role_id}/capabilities",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        # Not the caller's own tenant: authorization follows the role's owner,
        # so a caller cannot grant into a tenant they do not administer.
        assert authorized_against == [str(owning_tenant_id)]
        assert granted == [(str(role_id), "manage_users")]

    def test_revoke_looks_the_role_up_by_id_and_authorizes_against_its_tenant(self, client):
        role_id = uuid4()
        owning_tenant_id = uuid4()
        authorized_against = []
        revoked = []

        async def get_role_by_id(requested_role_id):
            return SimpleNamespace(id=role_id, tenant_id=owning_tenant_id)

        async def has_management(requester_id, tenant_id):
            authorized_against.append(str(tenant_id))
            return True

        async def revoke(target_role_id, permission_name):
            revoked.append((str(target_role_id), permission_name))

        with (
            _stub("get_role_by_id", get_role_by_id),
            _stub("has_user_management_permission", has_management),
            _stub("revoke_default_permission_from_role", revoke),
        ):
            response = client.request(
                "DELETE",
                f"{_PREFIX}/roles/{role_id}/capabilities",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 200
        assert authorized_against == [str(owning_tenant_id)]
        assert revoked == [(str(role_id), "manage_users")]

    def test_an_unknown_role_is_a_404(self, client):
        async def missing(role_id):
            raise RoleNotFoundError(message=f"Could not find role: {role_id}")

        async def has_management(requester_id, tenant_id):
            raise AssertionError("must not authorize against a role that does not exist")

        with (
            _stub("get_role_by_id", missing),
            _stub("has_user_management_permission", has_management),
        ):
            response = client.post(
                f"{_PREFIX}/roles/{uuid4()}/capabilities",
                params={"capability": "manage_users"},
            )

        assert response.status_code == 404
