from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.api.DTO import InDTO


class DatasetPermissionsPayloadDTO(InDTO):
    principal_id: str
    permission_name: str
    dataset_ids: List[str]


def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/tenants/{role_id}/default_permissions")
    async def give_default_permission_to_role(
        role_id: UUID, permission_name: str, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.users.permissions.methods import (
            give_default_permission_to_role as set_default_permission_to_role,
        )

        await set_default_permission_to_role(role_id, permission_name)

        return JSONResponse(status_code=200, content={"message": "Permission assigned to role"})

    @permissions_router.post("/tenants/{tenant_id}/default_permissions")
    async def give_default_permission_to_tenant(
        tenant_id: UUID, permission_name: str, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.users.permissions.methods import (
            give_default_permission_to_tenant as set_tenant_default_permissions,
        )

        await set_tenant_default_permissions(tenant_id, permission_name)

        return JSONResponse(status_code=200, content={"message": "Permission assigned to tenant"})

    @permissions_router.post("/tenants/{user_id}/default_permissions")
    async def give_default_permission_to_user(
        user_id: UUID, permission_name: str, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.users.permissions.methods import (
            give_default_permission_to_user as set_default_permission_to_user,
        )

        await set_default_permission_to_user(user_id, permission_name)

        return JSONResponse(status_code=200, content={"message": "Permission assigned to user"})

    @permissions_router.post("/datasets/{principal_id}/")
    async def give_datasets_permission_to_principal(
        payload: DatasetPermissionsPayloadDTO, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets

        await authorized_give_permission_on_datasets(
            UUID(payload.principal_id),
            [UUID(dataset_id) for dataset_id in payload.dataset_ids],
            payload.permission_name,
            user.id,
        )

        return JSONResponse(status_code=200, content={"message": "Permission assigned to user"})

    @permissions_router.post("/roles")
    async def create_role(role_name: str, user: User = Depends(get_authenticated_user)):
        from cognee.modules.users.roles.methods import create_role as create_role_method

        await create_role_method(role_name=role_name, tenant_id=user.tenant_id)

        return JSONResponse(status_code=200, content={"message": "Role created for tenant"})

    @permissions_router.post("/users/{user_id}/roles")
    async def add_user_to_role(
        user_id: str, role_id: str, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.users.roles.methods import add_user_to_role as add_user_to_role_method

        await add_user_to_role_method(user_id=UUID(user_id), role_id=UUID(role_id))

        return JSONResponse(status_code=200, content={"message": "User added to role"})

    @permissions_router.post("/users/{user_id}/tenants")
    async def add_user_to_tenant(user_id: str, user: User = Depends(get_authenticated_user)):
        from cognee.modules.users.tenants.methods import add_user_to_tenant

        await add_user_to_tenant(user_id=UUID(user_id), tenant_id=user.tenant_id, owner_id=user.id)

        return JSONResponse(status_code=200, content={"message": "User added to tenant"})

    @permissions_router.post("/tenants")
    async def create_tenant(tenant_name: str, user: User = Depends(get_authenticated_user)):
        from cognee.modules.users.tenants.methods import create_tenant as create_tenant_method

        await create_tenant_method(tenant_name=tenant_name, user_id=user.id)

        return JSONResponse(status_code=200, content={"message": "Tenant created."})

    return permissions_router
