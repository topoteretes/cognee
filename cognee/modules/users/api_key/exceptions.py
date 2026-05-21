class ApiKeyCreationError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "Failed to create API key."


class ApiKeyDeletionError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "Failed to delete API key."


class ApiKeyQueryError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "Failed to query API keys."


class ApiKeyMissingError(Exception):
    def __init__(self, message: str):
        self.message = message

    message: str = "No available API keys."
