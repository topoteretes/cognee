from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import JSONResponse


def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/roles/{role_id}/permissions")
    async def give_default_permission_to_role(role_id: UUID, permission_name: str):
        from cognee.modules.users.permissions.methods import (
            give_default_permission_to_role as set_default_permission_to_role,
        )

        await set_default_permission_to_role(role_id, permission_name)

        return JSONResponse(status_code=200, content={"message": "Permission assigned to role"})

    @permissions_router.post("/tenants/{tenant_id}/permissions")
    async def give_default_permission_to_tenant(tenant_id: UUID, permission_name: str):
        from cognee.modules.users.permissions.methods import (
            give_default_permission_to_tenant as set_tenant_default_permissions,
        )

        await set_tenant_default_permissions(tenant_id, permission_name)

        return JSONResponse(status_code=200, content={"message": "Permission assigned to tenant"})

    @permissions_router.post("/users/{user_id}/permissions")
    async def give_default_permission_to_user(user_id: UUID, permission_name: str):
        from cognee.modules.users.permissions.methods import (
            give_default_permission_to_user as set_default_permission_to_user,
        )

        await set_default_permission_to_user(user_id, permission_name)

        return JSONResponse(status_code=200, content={"message": "Permission assigned to user"})

    @permissions_router.post("/roles")
    async def create_role(
        role_name: str,
        tenant_id: UUID,
    ):
        from cognee.modules.users.roles.methods import create_role as create_role_method

        await create_role_method(role_name=role_name, tenant_id=tenant_id)

        return JSONResponse(status_code=200, content={"message": "Role created for tenant"})

    @permissions_router.post("/users/{user_id}/roles")
    async def add_user_to_role(user_id: UUID, role_id: UUID):
        from cognee.modules.users.roles.methods import add_user_to_role as add_user_to_role_method

        await add_user_to_role_method(user_id=user_id, role_id=role_id)

        return JSONResponse(status_code=200, content={"message": "User added to role"})

    @permissions_router.post("/tenants")
    async def create_tenant(tenant_name: str):
        from cognee.modules.users.tenants.methods import create_tenant as create_tenant_method

        await create_tenant_method(tenant_name=tenant_name)

        return JSONResponse(status_code=200, content={"message": "Tenant created."})

    return permissions_router
