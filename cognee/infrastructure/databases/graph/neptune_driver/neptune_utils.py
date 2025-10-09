"""Neptune Utilities

This module provides utility functions for Neptune Analytics operations including
connection management, URL parsing, and Neptune-specific configurations.
"""

import re
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

from cognee.shared.logging_utils import get_logger

logger = get_logger("NeptuneUtils")


def parse_neptune_url(url: str) -> Tuple[str, str]:
    """
    Parse a Neptune Analytics URL to extract graph ID and region.

    Expected format: neptune-graph://<GRAPH_ID>?region=<REGION>
    or neptune-graph://<GRAPH_ID> (defaults to us-east-1)

    Parameters:
    -----------
        - url (str): The Neptune Analytics URL to parse

    Returns:
    --------
        - Tuple[str, str]: A tuple containing (graph_id, region)

    Raises:
    -------
        - ValueError: If the URL format is invalid
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme != "neptune-graph":
            raise ValueError(f"Invalid scheme: {parsed.scheme}. Expected 'neptune-graph'")

        graph_id = parsed.hostname or parsed.path.lstrip("/")
        if not graph_id:
            raise ValueError("Graph ID not found in URL")

        # Extract region from query parameters
        region = "us-east-1"  # default region
        if parsed.query:
            query_params = dict(
                param.split("=") for param in parsed.query.split("&") if "=" in param
            )
            region = query_params.get("region", region)

        return graph_id, region

    except Exception as e:
        raise ValueError(f"Failed to parse Neptune Analytics URL '{url}': {str(e)}") from e


def validate_graph_id(graph_id: str) -> bool:
    """
    Validate a Neptune Analytics graph ID format.

    Graph IDs should follow AWS naming conventions.

    Parameters:
    -----------
        - graph_id (str): The graph ID to validate

    Returns:
    --------
        - bool: True if the graph ID is valid, False otherwise
    """
    if not graph_id:
        return False

    # Neptune Analytics graph IDs should be alphanumeric with hyphens
    # and between 1-63 characters
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9\-]{0,62}$"
    return bool(re.match(pattern, graph_id))


def validate_aws_region(region: str) -> bool:
    """
    Validate an AWS region format.

    Parameters:
    -----------
        - region (str): The AWS region to validate

    Returns:
    --------
        - bool: True if the region format is valid, False otherwise
    """
    if not region:
        return False

    # AWS regions follow the pattern: us-east-1, eu-west-1, etc.
    pattern = r"^[a-z]{2,3}-[a-z]+-\d+$"
    return bool(re.match(pattern, region))


def build_neptune_config(
    graph_id: str,
    region: Optional[str],
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build a configuration dictionary for Neptune Analytics connection.

    Parameters:
    -----------
        - graph_id (str): The Neptune Analytics graph identifier
        - region (Optional[str]): AWS region where the graph is located
        - aws_access_key_id (Optional[str]): AWS access key ID
        - aws_secret_access_key (Optional[str]): AWS secret access key
        - aws_session_token (Optional[str]): AWS session token for temporary credentials
        - **kwargs: Additional configuration parameters

    Returns:
    --------
        - Dict[str, Any]: Configuration dictionary for Neptune Analytics

    Raises:
    -------
        - ValueError: If required parameters are invalid
    """
    config = {
        "graph_id": graph_id,
        "service_name": "neptune-graph",
    }

    # Add AWS credentials if provided
    if region:
        config["region"] = region

    if aws_access_key_id:
        config["aws_access_key_id"] = aws_access_key_id

    if aws_secret_access_key:
        config["aws_secret_access_key"] = aws_secret_access_key

    if aws_session_token:
        config["aws_session_token"] = aws_session_token

    # Add any additional configuration
    config.update(kwargs)

    return config


def get_neptune_endpoint_url(graph_id: str, region: str) -> str:
    """
    Construct the Neptune Analytics endpoint URL for a given graph and region.

    Parameters:
    -----------
        - graph_id (str): The Neptune Analytics graph identifier
        - region (str): AWS region where the graph is located

    Returns:
    --------
        - str: The Neptune Analytics endpoint URL
    """
    return f"https://neptune-graph.{region}.amazonaws.com/graphs/{graph_id}"


def format_neptune_error(error: Exception) -> str:
    """
    Format Neptune Analytics specific errors for better readability.

    Parameters:
    -----------
        - error (Exception): The exception to format

    Returns:
    --------
        - str: Formatted error message
    """
    error_msg = str(error)

    # Common Neptune Analytics error patterns and their user-friendly messages
    error_mappings = {
        "AccessDenied": "Access denied. Please check your AWS credentials and permissions.",
        "GraphNotFound": "Graph not found. Please verify the graph ID and region.",
        "InvalidParameter": "Invalid parameter provided. Please check your request parameters.",
        "ThrottlingException": "Request was throttled. Please retry with exponential backoff.",
        "InternalServerError": "Internal server error occurred. Please try again later.",
    }

    for error_type, friendly_msg in error_mappings.items():
        if error_type in error_msg:
            return f"{friendly_msg} Original error: {error_msg}"

    return error_msg


def get_default_query_timeout() -> int:
    """
    Get the default query timeout for Neptune Analytics operations.

    Returns:
    --------
        - int: Default timeout in seconds
    """
    return 300  # 5 minutes


def get_default_connection_config() -> Dict[str, Any]:
    """
    Get default connection configuration for Neptune Analytics.

    Returns:
    --------
        - Dict[str, Any]: Default connection configuration
    """
    return {
        "query_timeout": get_default_query_timeout(),
        "max_retries": 3,
        "retry_delay": 1.0,
        "preferred_query_language": "openCypher",
    }
