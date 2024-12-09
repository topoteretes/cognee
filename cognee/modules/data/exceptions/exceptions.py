from cognee.exceptions import CogneeApiError
from fastapi import status

class UnstructuredLibraryImportError(CogneeApiError):
    def __init__(
            self,
            message: str = "Import error. Unstructured library is not installed.",
            name: str = "UnstructuredModuleImportError",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    ):
        super().__init__(message, name, status_code)