from fastapi import status

from cognee.exceptions import CogneeConfigurationError, CogneeValidationError


class InvalidConfigAttributeError(CogneeConfigurationError):
    def __init__(
        self,
        attribute: str,
        name: str = "InvalidConfigAttributeError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = f"'{attribute}' is not a valid attribute of the configuration."
        super().__init__(message, name, status_code)


class DocumentNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Document not found in database.",
        name: str = "DocumentNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class DatasetNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Dataset not found.",
        name: str = "DatasetNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class DataNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Data not found.",
        name: str = "DataNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class UpdateTargetNotFoundError(CogneeValidationError):
    """update() was called with a data_id that resolves to no document.

    Neither an exact row id nor a recorded pre-fork ``legacy_id`` matched in
    the dataset. update() replaces existing documents only — use add() to
    create new ones.
    """

    def __init__(
        self,
        data_id,
        dataset_id,
        name: str = "UpdateTargetNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        message = (
            f"No document found to update: data_id '{data_id}' does not exist "
            f"in dataset '{dataset_id}' (neither as a document id nor as a "
            f"pre-fork legacy id). Use add() to create new documents."
        )
        super().__init__(message, name, status_code)


class DocumentUpdateRequiredError(CogneeValidationError):
    """add() was given a file that already exists in the dataset with other content.

    add() creates documents and leaves existing ones alone: a file that
    matches a stored document by origin (the same path, or the same filename
    for an upload) but carries different content is an update, and updates
    go through update() so the document keeps its id and its graph is
    replaced in place instead of a second copy being minted. Identical
    content is not an error: re-adding it is a no-op.

    ``conflicts`` lists every offending file as ``{"name", "data_id"}`` and
    ``api_message`` says the same thing for HTTP callers, pointing at
    ``PATCH /api/v1/update``.
    """

    def __init__(
        self,
        conflicts: list[dict],
        dataset_id,
        name: str = "DocumentUpdateRequiredError",
        status_code: int = status.HTTP_409_CONFLICT,
    ):
        self.conflicts = conflicts
        self.dataset_id = dataset_id
        super().__init__(self._describe(sdk=True), name, status_code)

    @property
    def api_message(self) -> str:
        """The same refusal for HTTP callers, pointing at the update endpoint."""
        return self._describe(sdk=False)

    def _describe(self, sdk: bool) -> str:
        listed = ", ".join(f"'{c['name']}' (data_id {c['data_id']})" for c in self.conflicts)
        # With one offending file the pointer is the exact call to make.
        data_id = self.conflicts[0]["data_id"] if len(self.conflicts) == 1 else "<data_id>"
        verb = "exists" if len(self.conflicts) == 1 else "exist"
        if sdk:
            return (
                f"add() does not update documents. {listed} already {verb} in dataset "
                f"{self.dataset_id} with different content. To replace the stored version, call "
                f"cognee.update(data_id={data_id}, data=<new content>, dataset_id={self.dataset_id})"
                f"{'' if len(self.conflicts) == 1 else ' for each document'}; identical content "
                "can be re-added and is a no-op."
            )
        return (
            f"POST /api/v1/add does not update documents. {listed} already {verb} in "
            f"dataset {self.dataset_id} with different content. Send the new version to "
            f"PATCH /api/v1/update?data_id={data_id}&dataset_id={self.dataset_id} (multipart "
            "field 'data'), or cognee.update(...) from the SDK; identical content can be "
            "re-added and is a no-op."
        )


class DocumentSubgraphNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Document subgraph not found in graph database.",
        name: str = "DocumentSubgraphNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)
