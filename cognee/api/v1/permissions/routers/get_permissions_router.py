from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version


def get_permissions_router() -> APIRouter:
    permissions_router = APIRouter()

    @permissions_router.post("/datasets/{principal_id}")
    async def give_datasets_permission_to_principal(
        permission_name: str,
        dataset_ids: List[UUID],
        principal_id: UUID,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Grant permission on datasets to a principal (user or role).

        This endpoint allows granting specific permissions on one or more datasets
        to a principal (which can be a user or role). The authenticated user must
        have appropriate permissions to grant access to the specified datasets.

        ## Path Parameters
        - **principal_id** (UUID): The UUID of the principal (user or role) to grant permission to

        ## Request Parameters
        - **permission_name** (str): The name of the permission to grant (e.g., "read", "write", "delete")
        - **dataset_ids** (List[UUID]): List of dataset UUIDs to grant permission on

        ## Response
        Returns a success message indicating permission was assigned.

        ## Error Codes
        - **400 Bad Request**: Invalid request parameters
        - **403 Forbidden**: User doesn't have permission to grant access
        - **500 Internal Server Error**: Error granting permission
        """
        send_telemetry(
            "Permissions API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"POST /v1/permissions/datasets/{str(principal_id)}",
                "dataset_ids": str(dataset_ids),
                "principal_id": str(principal_id),
                "cognee_version": cognee_version,
            },
        )

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
        """
        Create a new role.

        This endpoint creates a new role with the specified name. Roles are used
        to group permissions and can be assigned to users to manage access control
        more efficiently. The authenticated user becomes the owner of the created role.

        ## Request Parameters
        - **role_name** (str): The name of the role to create

        ## Response
        Returns a success message indicating the role was created.

        ## Error Codes
        - **400 Bad Request**: Invalid role name or role already exists
        - **500 Internal Server Error**: Error creating the role
        """
        send_telemetry(
            "Permissions API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/permissions/roles",
                "role_name": role_name,
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.users.roles.methods import create_role as create_role_method

        role_id = await create_role_method(role_name=role_name, owner_id=user.id)

        return JSONResponse(
            status_code=200, content={"message": "Role created for tenant", "role_id": str(role_id)}
        )

    @permissions_router.post("/users/{user_id}/roles")
    async def add_user_to_role(
        user_id: UUID, role_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        """
        Add a user to a role.

        This endpoint assigns a user to a specific role, granting them all the
        permissions associated with that role. The authenticated user must be
        the owner of the role or have appropriate administrative permissions.

        ## Path Parameters
        - **user_id** (UUID): The UUID of the user to add to the role

        ## Request Parameters
        - **role_id** (UUID): The UUID of the role to assign the user to

        ## Response
        Returns a success message indicating the user was added to the role.

        ## Error Codes
        - **400 Bad Request**: Invalid user or role ID
        - **403 Forbidden**: User doesn't have permission to assign roles
        - **404 Not Found**: User or role doesn't exist
        - **500 Internal Server Error**: Error adding user to role
        """
        send_telemetry(
            "Permissions API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"POST /v1/permissions/users/{str(user_id)}/roles",
                "user_id": str(user_id),
                "role_id": str(role_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.users.roles.methods import add_user_to_role as add_user_to_role_method

        await add_user_to_role_method(user_id=user_id, role_id=role_id, owner_id=user.id)

        return JSONResponse(status_code=200, content={"message": "User added to role"})

    @permissions_router.post("/users/{user_id}/tenants")
    async def add_user_to_tenant(
        user_id: UUID, tenant_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        """
        Add a user to a tenant.

        This endpoint assigns a user to a specific tenant, allowing them to access
        resources and data associated with that tenant. The authenticated user must
        be the owner of the tenant or have appropriate administrative permissions.

        ## Path Parameters
        - **user_id** (UUID): The UUID of the user to add to the tenant

        ## Request Parameters
        - **tenant_id** (UUID): The UUID of the tenant to assign the user to

        ## Response
        Returns a success message indicating the user was added to the tenant.

        ## Error Codes
        - **400 Bad Request**: Invalid user or tenant ID
        - **403 Forbidden**: User doesn't have permission to assign tenants
        - **404 Not Found**: User or tenant doesn't exist
        - **500 Internal Server Error**: Error adding user to tenant
        """
        send_telemetry(
            "Permissions API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"POST /v1/permissions/users/{str(user_id)}/tenants",
                "user_id": str(user_id),
                "tenant_id": str(tenant_id),
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.users.tenants.methods import add_user_to_tenant

        await add_user_to_tenant(user_id=user_id, tenant_id=tenant_id, owner_id=user.id)

        return JSONResponse(status_code=200, content={"message": "User added to tenant"})

    @permissions_router.post("/tenants")
    async def create_tenant(tenant_name: str, user: User = Depends(get_authenticated_user)):
        """
        Create a new tenant.

        This endpoint creates a new tenant with the specified name. Tenants are used
        to organize users and resources in multi-tenant environments, providing
        isolation and access control between different groups or organizations.

        ## Request Parameters
        - **tenant_name** (str): The name of the tenant to create

        ## Response
        Returns a success message indicating the tenant was created.

        ## Error Codes
        - **400 Bad Request**: Invalid tenant name or tenant already exists
        - **500 Internal Server Error**: Error creating the tenant
        """
        send_telemetry(
            "Permissions API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/permissions/tenants",
                "tenant_name": tenant_name,
                "cognee_version": cognee_version,
            },
        )

        from cognee.modules.users.tenants.methods import create_tenant as create_tenant_method

        tenant_id = await create_tenant_method(tenant_name=tenant_name, user_id=user.id)

        return JSONResponse(
            status_code=200, content={"message": "Tenant created.", "tenant_id": str(tenant_id)}
        )

    return permissions_router
