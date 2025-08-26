from typing import Optional


class CliCommandException(Exception):
    """Exception raised by CLI commands with additional context"""

    def __init__(
        self,
        message: str,
        error_code: int = -1,
        docs_url: Optional[str] = None,
        raiseable_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.docs_url = docs_url
        self.raiseable_exception = raiseable_exception


class CliCommandInnerException(Exception):
    """Inner exception for wrapping other exceptions in CLI context"""

    pass
