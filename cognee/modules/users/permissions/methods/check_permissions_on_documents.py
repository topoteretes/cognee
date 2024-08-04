import  logging
from cognee.infrastructure.databases.relational import get_relational_engine
from ...models.User import User
from ...models.ACL import ACL

logger = logging.getLogger(__name__)

async def check_permissions_on_documents(
    user: User,
    permission_type: str,
    document_ids: list[str],
):
    try:
        relational_engine = get_relational_engine()

        async with relational_engine.get_async_session() as session:
            user_group_ids = [group.id for group in user.groups]

            acls = session.query(ACL) \
                .filter(ACL.principal_id.in_([user.id, *user_group_ids])) \
                .filter(ACL.permission.name == permission_type) \
                .all()

            resource_ids = [resource.resource_id for resource in acl.resources for acl in acls]

            has_permissions = all([document_id in resource_ids for document_id in document_ids])

            if not has_permissions:
                raise Exception(f"User {user.username} does not have {permission_type} permission on documents")
    except Exception as error:
        logger.error("Error checking permissions on documents: %s", str(error))
        raise error
