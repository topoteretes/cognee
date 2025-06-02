from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user


def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/datasets/{principal_id}/")
    async def give_datasets_permission_to_principal(
        permission_name: str,
        dataset_ids: List[UUID],
        principal_id: UUID,
        user: User = Depends(get_authenticated_user),
    ):
        from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets

        await authorized_give_permission_on_datasets(
            principal_id,
            [dataset_id for dataset_id in dataset_ids],
            permission_name,
            user.id,
        )

        return JSONResponse(
            status_code=200, content={"message": "Permission assigned to principal"}
        )

    @permissions_router.post("/roles")
    async def create_role(role_name: str, user: User = Depends(get_authenticated_user)):
        from cognee.modules.users.roles.methods import create_role as create_role_method

        await create_role_method(role_name=role_name, owner_id=user.id)

        return JSONResponse(status_code=200, content={"message": "Role created for tenant"})

    @permissions_router.post("/users/{user_id}/roles")
    async def add_user_to_role(
        user_id: UUID, role_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.users.roles.methods import add_user_to_role as add_user_to_role_method

        await add_user_to_role_method(user_id=user_id, role_id=role_id, owner_id=user.id)

        return JSONResponse(status_code=200, content={"message": "User added to role"})

    @permissions_router.post("/users/{user_id}/tenants")
    async def add_user_to_tenant(
        user_id: UUID, tenant_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        from cognee.modules.users.tenants.methods import add_user_to_tenant

        await add_user_to_tenant(user_id=user_id, tenant_id=tenant_id, owner_id=user.id)

        return JSONResponse(status_code=200, content={"message": "User added to tenant"})

    @permissions_router.post("/tenants")
    async def create_tenant(tenant_name: str, user: User = Depends(get_authenticated_user)):
        from cognee.modules.users.tenants.methods import create_tenant as create_tenant_method

        await create_tenant_method(tenant_name=tenant_name, user_id=user.id)

        return JSONResponse(status_code=200, content={"message": "Tenant created."})

    return permissions_router
