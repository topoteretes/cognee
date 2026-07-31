import pytest

from cognee.exceptions import (
    CogneeConfigurationError,
    CogneeTransientError,
    CogneeValidationError,
    ErrorCode,
    serialize_cognee_error,
)
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.modules.users.exceptions.exceptions import PermissionDeniedError

REQUIRED_ENVELOPE_KEYS = frozenset(
    {
        "schema_version",
        "code",
        "message",
        "name",
        "remediation",
        "retryable",
        "status_code",
    }
)


def test_envelope_required_keys():
    env = serialize_cognee_error(CogneeValidationError("x"))
    assert REQUIRED_ENVELOPE_KEYS <= frozenset(env.keys())


def test_schema_version_is_1():
    env = serialize_cognee_error(CogneeValidationError("x"))
    assert env["schema_version"] == 1


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (NoDataError(stage="add"), ErrorCode.DATA_NOT_READY),
        (CogneeConfigurationError("missing key"), ErrorCode.MISSING_CONFIG),
        (CogneeTransientError(), ErrorCode.TRANSIENT),
        (UnauthorizedDataAccessError(), ErrorCode.PERMISSION_DENIED),
        (LLMAPIKeyNotSetError(), ErrorCode.MISSING_CONFIG),
        (PermissionDeniedError(), ErrorCode.PERMISSION_DENIED),
        (CogneeValidationError("bad"), ErrorCode.INVALID_INPUT),
    ],
)
def test_golden_code_per_category(exc, expected_code):
    env = serialize_cognee_error(exc)
    assert env["code"] == expected_code.value
