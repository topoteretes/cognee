from uuid import UUID
from sqlalchemy import select
from cognee.infrastructure.databases.relational.get_async_session import get_async_session
from cognee.modules.users.models.ACL import ACL
from cognee.modules.users.models.Permission import Permission
from cognee.modules.users.models.Role import Role
from cognee.modules.users.models.Tenant import Tenant
from cognee.modules.users.models.UserTenant import UserTenant
from cognee.modules.users.models.RoleDefaultPermissions import RoleDefaultPermissions
from cognee.modules.users.models.TenantDefaultPermissions import TenantDefaultPermissions
from cognee.modules.users.models.UserDefaultPermissions import UserDefaultPermissions

from cognee.modules.governance.models import ODRLPolicy

COGNEE_TO_ODRL_ACTION = {
    "read":   "https://www.w3.org/ns/odrl/2/read",
    "write":  "https://www.w3.org/ns/odrl/2/modify",
    "delete": "https://www.w3.org/ns/odrl/2/delete",
    "admin":  "https://www.w3.org/ns/odrl/2/All",
}

async def serialize_permission_model(dataset_id: UUID) -> list[ODRLPolicy]:
    """
    Reads ACL, Permission, Role, Tenant, UserTenant,
    RoleDefaultPermissions, TenantDefaultPermissions, UserDefaultPermissions
    tables for the given dataset_id and converts each row to an ODRLPolicy.
    """
    policies = []
    
    async with get_async_session() as session:
        # Load ACLs with permissions for this dataset
        stmt = (
            select(ACL, Permission)
            .join(Permission, ACL.permission_id == Permission.id)
            .where(ACL.dataset_id == dataset_id)
        )
        result = await session.execute(stmt)
        acl_rows = result.all()
        
        for acl_row, perm_row in acl_rows:
            action_uri = COGNEE_TO_ODRL_ACTION.get(perm_row.name, f"urn:cognee:action:{perm_row.name}")
            
            policies.append(ODRLPolicy(
                uid=str(acl_row.id),
                type="Set",
                assigner="urn:cognee:system",
                assignee=f"urn:cognee:principal:{acl_row.principal_id}",
                target=f"urn:cognee:dataset:{dataset_id}",
                action=action_uri,
                custom_action=perm_row.name if perm_row.name not in COGNEE_TO_ODRL_ACTION else None
            ))
            
        # RoleDefaultPermissions
        rdp_stmt = select(RoleDefaultPermissions, Permission, Role).join(Permission, RoleDefaultPermissions.permission_id == Permission.id).join(Role, RoleDefaultPermissions.role_id == Role.id)
        rdp_result = await session.execute(rdp_stmt)
        for rdp_row, perm_row, role_row in rdp_result.all():
            action_uri = COGNEE_TO_ODRL_ACTION.get(perm_row.name, f"urn:cognee:action:{perm_row.name}")
            policies.append(ODRLPolicy(
                uid=f"rdp_{rdp_row.role_id}_{rdp_row.permission_id}",
                type="Set",
                assigner=f"urn:cognee:tenant:{role_row.tenant_id}",
                assignee=f"urn:cognee:role:{rdp_row.role_id}",
                target=f"urn:cognee:dataset:{dataset_id}",
                action=action_uri,
                custom_action=perm_row.name if perm_row.name not in COGNEE_TO_ODRL_ACTION else None
            ))
            
        # TenantDefaultPermissions
        tdp_stmt = select(TenantDefaultPermissions, Permission, Tenant).join(Permission, TenantDefaultPermissions.permission_id == Permission.id).join(Tenant, TenantDefaultPermissions.tenant_id == Tenant.id)
        tdp_result = await session.execute(tdp_stmt)
        for tdp_row, perm_row, tenant_row in tdp_result.all():
            action_uri = COGNEE_TO_ODRL_ACTION.get(perm_row.name, f"urn:cognee:action:{perm_row.name}")
            policies.append(ODRLPolicy(
                uid=f"tdp_{tdp_row.tenant_id}_{tdp_row.permission_id}",
                type="Set",
                assigner=f"urn:cognee:tenant:{tdp_row.tenant_id}",
                assignee=f"urn:cognee:tenant:{tdp_row.tenant_id}",
                target=f"urn:cognee:dataset:{dataset_id}",
                action=action_uri,
                custom_action=perm_row.name if perm_row.name not in COGNEE_TO_ODRL_ACTION else None
            ))
            
        # UserDefaultPermissions
        udp_stmt = select(UserDefaultPermissions, Permission).join(Permission, UserDefaultPermissions.permission_id == Permission.id)
        udp_result = await session.execute(udp_stmt)
        for udp_row, perm_row in udp_result.all():
            action_uri = COGNEE_TO_ODRL_ACTION.get(perm_row.name, f"urn:cognee:action:{perm_row.name}")
            policies.append(ODRLPolicy(
                uid=f"udp_{udp_row.user_id}_{udp_row.permission_id}",
                type="Set",
                assigner="urn:cognee:system",
                assignee=f"urn:cognee:user:{udp_row.user_id}",
                target=f"urn:cognee:dataset:{dataset_id}",
                action=action_uri,
                custom_action=perm_row.name if perm_row.name not in COGNEE_TO_ODRL_ACTION else None
            ))
            
        # UserTenant (implicitly grants some association)
        ut_stmt = select(UserTenant)
        ut_result = await session.execute(ut_stmt)
        for ut_row in ut_result.scalars().all():
            policies.append(ODRLPolicy(
                uid=f"ut_{ut_row.user_id}_{ut_row.tenant_id}",
                type="Set",
                assigner=f"urn:cognee:tenant:{ut_row.tenant_id}",
                assignee=f"urn:cognee:user:{ut_row.user_id}",
                target=f"urn:cognee:tenant:{ut_row.tenant_id}",
                action="urn:cognee:action:member",
                custom_action="member"
            ))
            
    return policies
