class TranslationError(Exception):
    """Base exception for translation errors."""

    def __init__(self, message: str, original_error: Exception = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)
        if original_error:
            self.__cause__ = original_error


class LanguageDetectionError(TranslationError):
    """Exception raised when language detection fails."""

    def __init__(
        self, message: str = "Failed to detect language", original_error: Exception = None
    ):
        super().__init__(message, original_error)


class TranslationProviderError(TranslationError):
    """Exception raised when the translation provider encounters an error."""

    def __init__(
        self,
        provider: str,
        message: str = "Translation provider error",
        original_error: Exception = None,
    ):
        self.provider = provider
        full_message = f"[{provider}] {message}"
        super().__init__(full_message, original_error)


class UnsupportedLanguageError(TranslationError):
    """Exception raised when the language is not supported."""

    def __init__(
        self,
        language: str,
        provider: str = None,
        message: str = None,
        original_error: Exception = None,
    ):
        self.language = language
        self.provider = provider
        if message is None:
            message = f"Language '{language}' is not supported"
            if provider:
                message += f" by {provider}"
        super().__init__(message, original_error)


class TranslationConfigError(TranslationError):
    """Exception raised when translation configuration is invalid."""

    def __init__(
        self,
        message: str = "Invalid translation configuration",
        original_error: Exception = None,
    ):
        super().__init__(message, original_error)
