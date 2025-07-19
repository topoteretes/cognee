from typing import List, Optional, Dict, Any
from .enhanced_exceptions import (
    CogneeUserError,
    CogneeSystemError,
    CogneeTransientError,
    CogneeConfigurationError,
    CogneeValidationError,
    CogneeNotFoundError,
    CogneePermissionError,
)


# ========== DATA/INPUT ERRORS (User-fixable) ==========


class UnsupportedFileFormatError(CogneeValidationError):
    """File format not supported by Cognee"""

    def __init__(self, file_path: str, supported_formats: List[str], **kwargs):
        super().__init__(
            message=f"File format not supported: {file_path}",
            user_message=f"The file '{file_path}' has an unsupported format.",
            suggestions=[
                f"Use one of these supported formats: {', '.join(supported_formats)}",
                "Convert your file to a supported format",
                "Check our documentation for the complete list of supported formats",
            ],
            docs_link="https://docs.cognee.ai/guides/file-formats",
            context={"file_path": file_path, "supported_formats": supported_formats},
            operation="add",
            **kwargs,
        )


class EmptyDatasetError(CogneeValidationError):
    """Dataset is empty or contains no processable content"""

    def __init__(self, dataset_name: str, **kwargs):
        super().__init__(
            message=f"Dataset '{dataset_name}' is empty",
            user_message=f"The dataset '{dataset_name}' contains no data to process.",
            suggestions=[
                "Add some data to the dataset first using cognee.add()",
                "Check if your files contain readable text content",
                "Verify that your data was uploaded successfully",
            ],
            docs_link="https://docs.cognee.ai/guides/adding-data",
            context={"dataset_name": dataset_name},
            operation="cognify",
            **kwargs,
        )


class DatasetNotFoundError(CogneeNotFoundError):
    """Dataset not found or not accessible"""

    def __init__(
        self, dataset_identifier: str, available_datasets: Optional[List[str]] = None, **kwargs
    ):
        suggestions = ["Check the dataset name for typos"]
        if available_datasets:
            suggestions.extend(
                [
                    f"Available datasets: {', '.join(available_datasets)}",
                    "Use cognee.datasets() to see all your datasets",
                ]
            )
        else:
            suggestions.append("Create the dataset first by adding data to it")

        super().__init__(
            message=f"Dataset not found: {dataset_identifier}",
            user_message=f"Could not find dataset '{dataset_identifier}'.",
            suggestions=suggestions,
            docs_link="https://docs.cognee.ai/guides/datasets",
            context={
                "dataset_identifier": dataset_identifier,
                "available_datasets": available_datasets,
            },
            **kwargs,
        )


class InvalidQueryError(CogneeValidationError):
    """Search query is invalid or malformed"""

    def __init__(self, query: str, reason: str, **kwargs):
        super().__init__(
            message=f"Invalid query: {reason}",
            user_message=f"Your search query '{query}' is invalid: {reason}",
            suggestions=[
                "Try rephrasing your query",
                "Use simpler, more specific terms",
                "Check our query examples in the documentation",
            ],
            docs_link="https://docs.cognee.ai/guides/search",
            context={"query": query, "reason": reason},
            operation="search",
            **kwargs,
        )


class FileAccessError(CogneeUserError):
    """Cannot access or read the specified file"""

    def __init__(self, file_path: str, reason: str, **kwargs):
        super().__init__(
            message=f"Cannot access file: {file_path} - {reason}",
            user_message=f"Unable to read the file '{file_path}': {reason}",
            suggestions=[
                "Check if the file exists at the specified path",
                "Verify you have read permissions for the file",
                "Ensure the file is not locked by another application",
            ],
            context={"file_path": file_path, "reason": reason},
            operation="add",
            **kwargs,
        )


# ========== PROCESSING ERRORS (System/LLM errors) ==========


class LLMConnectionError(CogneeTransientError):
    """LLM service connection failure"""

    def __init__(self, provider: str, model: str, reason: str, **kwargs):
        super().__init__(
            message=f"LLM connection failed: {provider}/{model} - {reason}",
            user_message=f"Cannot connect to the {provider} language model service.",
            suggestions=[
                "Check your internet connection",
                "Verify your API key is correct and has sufficient credits",
                "Try again in a few moments",
                "Check the service status page",
            ],
            docs_link="https://docs.cognee.ai/troubleshooting/llm-connection",
            context={"provider": provider, "model": model, "reason": reason},
            **kwargs,
        )


class LLMRateLimitError(CogneeTransientError):
    """LLM service rate limit exceeded"""

    def __init__(self, provider: str, retry_after: Optional[int] = None, **kwargs):
        suggestions = [
            "Wait a moment before retrying",
            "Consider upgrading your API plan",
            "Use smaller batch sizes to reduce token usage",
        ]
        if retry_after:
            suggestions.insert(0, f"Wait {retry_after} seconds before retrying")

        super().__init__(
            message=f"Rate limit exceeded for {provider}",
            user_message=f"You've exceeded the rate limit for {provider}.",
            suggestions=suggestions,
            context={"provider": provider, "retry_after": retry_after},
            **kwargs,
        )


class ProcessingTimeoutError(CogneeTransientError):
    """Processing operation timed out"""

    def __init__(self, operation: str, timeout_seconds: int, **kwargs):
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            user_message=f"The {operation} operation took too long and was cancelled.",
            suggestions=[
                "Try processing smaller amounts of data at a time",
                "Check your internet connection stability",
                "Retry the operation",
                "Use background processing for large datasets",
            ],
            context={"operation": operation, "timeout_seconds": timeout_seconds},
            **kwargs,
        )


class DatabaseConnectionError(CogneeSystemError):
    """Database connection failure"""

    def __init__(self, db_type: str, reason: str, **kwargs):
        super().__init__(
            message=f"{db_type} database connection failed: {reason}",
            user_message=f"Cannot connect to the {db_type} database.",
            suggestions=[
                "Check if the database service is running",
                "Verify database connection configuration",
                "Check network connectivity",
                "Contact support if the issue persists",
            ],
            docs_link="https://docs.cognee.ai/troubleshooting/database",
            context={"db_type": db_type, "reason": reason},
            **kwargs,
        )


class InsufficientResourcesError(CogneeSystemError):
    """System has insufficient resources to complete the operation"""

    def __init__(self, resource_type: str, required: str, available: str, **kwargs):
        super().__init__(
            message=f"Insufficient {resource_type}: need {required}, have {available}",
            user_message=f"Not enough {resource_type} available to complete this operation.",
            suggestions=[
                "Try processing smaller amounts of data",
                "Free up system resources",
                "Wait for other operations to complete",
                "Consider upgrading your system resources",
            ],
            context={"resource_type": resource_type, "required": required, "available": available},
            **kwargs,
        )


# ========== CONFIGURATION ERRORS ==========


class MissingAPIKeyError(CogneeConfigurationError):
    """Required API key is missing"""

    def __init__(self, service: str, env_var: str, **kwargs):
        super().__init__(
            message=f"Missing API key for {service}",
            user_message=f"API key for {service} is not configured.",
            suggestions=[
                f"Set the {env_var} environment variable",
                f"Add your {service} API key to your .env file",
                "Check the setup documentation for detailed instructions",
            ],
            docs_link="https://docs.cognee.ai/setup/api-keys",
            context={"service": service, "env_var": env_var},
            **kwargs,
        )


class InvalidDatabaseConfigError(CogneeConfigurationError):
    """Database configuration is invalid"""

    def __init__(self, db_type: str, config_issue: str, **kwargs):
        super().__init__(
            message=f"Invalid {db_type} database configuration: {config_issue}",
            user_message=f"The {db_type} database is not properly configured: {config_issue}",
            suggestions=[
                "Check your database configuration settings",
                "Verify connection strings and credentials",
                "Review the database setup documentation",
                "Ensure the database server is accessible",
            ],
            docs_link="https://docs.cognee.ai/setup/databases",
            context={"db_type": db_type, "config_issue": config_issue},
            **kwargs,
        )


class UnsupportedSearchTypeError(CogneeValidationError):
    """Search type is not supported"""

    def __init__(self, search_type: str, supported_types: List[str], **kwargs):
        super().__init__(
            message=f"Unsupported search type: {search_type}",
            user_message=f"The search type '{search_type}' is not supported.",
            suggestions=[
                f"Use one of these supported search types: {', '.join(supported_types)}",
                "Check the search documentation for available types",
                "Try using GRAPH_COMPLETION for general queries",
            ],
            docs_link="https://docs.cognee.ai/guides/search-types",
            context={"search_type": search_type, "supported_types": supported_types},
            operation="search",
            **kwargs,
        )


# ========== PIPELINE ERRORS ==========


class PipelineExecutionError(CogneeSystemError):
    """Pipeline execution failed"""

    def __init__(self, pipeline_name: str, task_name: str, error_details: str, **kwargs):
        super().__init__(
            message=f"Pipeline '{pipeline_name}' failed at task '{task_name}': {error_details}",
            user_message=f"Processing failed during the {task_name} step.",
            suggestions=[
                "Check the logs for more detailed error information",
                "Verify your data is in a supported format",
                "Try processing smaller amounts of data",
                "Contact support if the issue persists",
            ],
            context={
                "pipeline_name": pipeline_name,
                "task_name": task_name,
                "error_details": error_details,
            },
            **kwargs,
        )


class DataExtractionError(CogneeSystemError):
    """Failed to extract content from data"""

    def __init__(self, source: str, reason: str, **kwargs):
        super().__init__(
            message=f"Data extraction failed for {source}: {reason}",
            user_message=f"Could not extract readable content from '{source}'.",
            suggestions=[
                "Verify the file is not corrupted",
                "Try converting to a different format",
                "Check if the file contains readable text",
                "Use a supported file format",
            ],
            context={"source": source, "reason": reason},
            operation="add",
            **kwargs,
        )


class NoDataToProcessError(CogneeValidationError):
    """No data available to process"""

    def __init__(self, operation: str, **kwargs):
        super().__init__(
            message=f"No data available for {operation}",
            user_message=f"There's no data to process for the {operation} operation.",
            suggestions=[
                "Add some data first using cognee.add()",
                "Check if your previous data upload was successful",
                "Verify the dataset contains processable content",
            ],
            docs_link="https://docs.cognee.ai/guides/adding-data",
            context={"operation": operation},
            **kwargs,
        )
