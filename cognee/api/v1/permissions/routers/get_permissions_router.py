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
        """
        Grant permission on datasets to a principal (user or role).

        This endpoint allows granting specific permissions on one or more datasets
        to a principal (which can be a user or role). The authenticated user must
        have appropriate permissions to grant access to the specified datasets.

        Args:
            permission_name (str): The name of the permission to grant (e.g., "read", "write", "delete")
            dataset_ids (List[UUID]): List of dataset UUIDs to grant permission on
            principal_id (UUID): The UUID of the principal (user or role) to grant permission to
            user: The authenticated user granting the permission

        Returns:
            JSONResponse: Success message indicating permission was assigned

        Raises:
            HTTPException: If there's an error granting the permission
            PermissionDeniedError: If the user doesn't have permission to grant access
        """
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

        Args:
            role_name (str): The name of the role to create
            user: The authenticated user creating the role

        Returns:
            JSONResponse: Success message indicating the role was created

        Raises:
            HTTPException: If there's an error creating the role
            ValidationError: If the role name is invalid or already exists
        """
        from cognee.modules.users.roles.methods import create_role as create_role_method

        await create_role_method(role_name=role_name, owner_id=user.id)

        return JSONResponse(status_code=200, content={"message": "Role created for tenant"})

    @permissions_router.post("/users/{user_id}/roles")
    async def add_user_to_role(
        user_id: UUID, role_id: UUID, user: User = Depends(get_authenticated_user)
    ):
        """
        Add a user to a role.

        This endpoint assigns a user to a specific role, granting them all the
        permissions associated with that role. The authenticated user must be
        the owner of the role or have appropriate administrative permissions.

        Args:
            user_id (UUID): The UUID of the user to add to the role
            role_id (UUID): The UUID of the role to assign the user to
            user: The authenticated user performing the role assignment

        Returns:
            JSONResponse: Success message indicating the user was added to the role

        Raises:
            HTTPException: If there's an error adding the user to the role
            PermissionDeniedError: If the user doesn't have permission to assign roles
            ValidationError: If the user or role doesn't exist
        """
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

        Args:
            user_id (UUID): The UUID of the user to add to the tenant
            tenant_id (UUID): The UUID of the tenant to assign the user to
            user: The authenticated user performing the tenant assignment

        Returns:
            JSONResponse: Success message indicating the user was added to the tenant

        Raises:
            HTTPException: If there's an error adding the user to the tenant
            PermissionDeniedError: If the user doesn't have permission to assign tenants
            ValidationError: If the user or tenant doesn't exist
        """
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

        Args:
            tenant_name (str): The name of the tenant to create
            user: The authenticated user creating the tenant

        Returns:
            JSONResponse: Success message indicating the tenant was created

        Raises:
            HTTPException: If there's an error creating the tenant
            ValidationError: If the tenant name is invalid or already exists
        """
        from cognee.modules.users.tenants.methods import create_tenant as create_tenant_method

        await create_tenant_method(tenant_name=tenant_name, user_id=user.id)

        return JSONResponse(status_code=200, content={"message": "Tenant created."})

    return permissions_router
