from uuid import UUID
from typing import List
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.models import Role, UserRole
from cognee.modules.users.exceptions import TenantNotFoundError
from cognee.modules.users.permissions.methods import get_tenant


class RoleInfo:
    """just a clean way to pass around role data without dealing with ORM objects everywhere."""
    
    def __init__(self, role_id: UUID, name: str, description: str = "", user_count: int = 0):
        self.id = role_id
        self.name = name
        self.description = description
        self.user_count = user_count
    
    def dict(self):
        """converts this to a dict so we can actually send it as JSON."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "user_count": self.user_count,
        }


async def get_roles_in_tenant(tenant_id: UUID) -> List[RoleInfo]:
    """
    grabs all the roles in a tenant and counts how many users are in each one.
    
    i wrote this because we needed to show role info without exposing the whole 
    database model to the API. also counting users turned out to be more annoying 
    than expected with async SQLAlchemy, so here we are.
    
    Args:
        tenant_id: which tenant we're looking at
    
    Returns:
        list of RoleInfo objects with the role details and user counts
    
    Raises:
        TenantNotFoundError: when you pass in a tenant that doesn't exist
    """
    # make sure this tenant is actually real before we do anything
    await get_tenant(tenant_id)
    
    db_engine = get_relational_engine()
    
    async with db_engine.get_async_session() as session:
        # grab all roles for this tenant and eagerly load the users relationship
        # because hitting the DB multiple times for user counts would be stupid
        query = (
            select(Role)
            .where(Role.tenant_id == tenant_id)
            .options(joinedload(Role.users))
        )
        
        result = await session.execute(query)
        roles = result.unique().scalars().all()
        
        # build our RoleInfo objects with actual user counts
        role_infos = []
        for role in roles:
            # sqlalchemy async relationships are weird, gotta check if we need awaitable_attrs
            users = await role.awaitable_attrs.users if hasattr(role, 'awaitable_attrs') else role.users
            user_count = len(users) if users else 0
            
            # get description if it exists on the model, otherwise empty string
            description = getattr(role, 'description', '') or ''
            
            role_info = RoleInfo(
                role_id=role.id,
                name=role.name,
                description=description,
                user_count=user_count
            )
            role_infos.append(role_info)
        
        return role_infos