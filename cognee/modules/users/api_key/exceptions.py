class ApiKeyCreationError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "Failed to create API key."


class ApiKeyDeletionError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "Failed to delete API key."


class ApiKeyNotFoundError(ApiKeyDeletionError):
    """The user holds no key with that id. A subclass, so existing handlers still catch it."""


class ApiKeyQueryError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "Failed to query API keys."


class ApiKeyMissingError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "No available API keys."
