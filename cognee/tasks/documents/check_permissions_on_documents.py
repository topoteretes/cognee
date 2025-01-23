from cognee.modules.data.processing.document_types import Document
from cognee.modules.users.permissions.methods import check_permission_on_documents
from typing import List


async def check_permissions_on_documents(
    documents: list[Document], user, permissions
) -> List[Document]:
    """
    Validates a user's permissions on a list of documents.

    Notes:
        - This function assumes that `check_permission_on_documents` raises an exception if the permission check fails.
        - It is designed to validate multiple permissions in a sequential manner for the same set of documents.
        - Ensure that the `Document` and `user` objects conform to the expected structure and interfaces.
    """
    document_ids = [document.id for document in documents]

    for permission in permissions:
        await check_permission_on_documents(
            user,
            permission,
            document_ids,
        )

    return documents
