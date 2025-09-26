class TranslationDependencyError(ImportError):
    """Raised when a required translation dependency is missing."""

class LangDetectError(TranslationDependencyError):
    """LangDetect library required."""
    def __init__(self, message="langdetect is not installed. Please install it with `pip install langdetect`"):
        super().__init__(message)

class GoogleTranslateError(TranslationDependencyError):
    """GoogleTrans library required."""
    def __init__(self, message="googletrans is not installed. Please install it with `pip install googletrans==4.0.0-rc1`"):
        super().__init__(message)

class AzureTranslateError(TranslationDependencyError):
    """Azure Translate library required."""
    def __init__(self, message="azure-ai-translation-text is not installed. Please install it with `pip install azure-ai-translation-text`"):
        super().__init__(message)

class AzureConfigError(ValueError):
    """Azure configuration error."""
    def __init__(self, message="Azure Translate key (AZURE_TRANSLATE_KEY) is required."):
        super().__init__(message)

class UnknownProviderError(ValueError):
    """Unknown translation provider error."""
    def __init__(self, provider_name=None):
        if provider_name:
            message = f"Unknown translation provider: {provider_name}."
        else:
            message = "Unknown translation provider."
        super().__init__(message)
