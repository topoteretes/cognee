from cognee.exceptions.exceptions import CogneeApiError


def test_cognee_api_error_defaults_to_5xx_range():
    """Ensure base API error defaults to a 5xx range for proper server-error monitoring."""
    error = CogneeApiError()
    assert 500 <= error.status_code < 600
