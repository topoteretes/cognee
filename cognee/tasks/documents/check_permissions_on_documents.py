from cognee.modules.data.processing.document_types import Document
from cognee.modules.users.models import User
from cognee.modules.users.permissions.methods import check_permission_on_documents

async def check_permissions_on_documents(documents: list[Document], user: User, permissions: list[str]) -> list[Document]:
    document_ids = [document.id for document in documents]

    for permission in permissions:
        await check_permission_on_documents(
            user,
            permission,
            document_ids,
        )

    return documents
