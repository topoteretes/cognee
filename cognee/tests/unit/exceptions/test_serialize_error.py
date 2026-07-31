import pytest

from cognee.exceptions import (
    CogneeConfigurationError,
    CogneeTransientError,
    CogneeValidationError,
    ErrorCode,
    http_error_content,
    serialize_cognee_error,
)
from cognee.modules.data.exceptions.exceptions import (
    DatasetNotFoundError,
    UnauthorizedDataAccessError,
)
from cognee.modules.retrieval.exceptions.exceptions import NoDataError


def test_serialize_validation_error():
    exc = CogneeValidationError(message="bad param", name="BadParam")
    envelope = serialize_cognee_error(exc)

    assert envelope["code"] == ErrorCode.INVALID_INPUT.value
    assert envelope["retryable"] is False
    assert envelope["message"] == "bad param"
    assert "remediation" in envelope
    assert envelope["schema_version"] == 1


def test_serialize_transient_error_is_retryable():
    exc = CogneeTransientError()
    envelope = serialize_cognee_error(exc)

    assert envelope["code"] == ErrorCode.TRANSIENT.value
    assert envelope["retryable"] is True


def test_serialize_configuration_error():
    exc = CogneeConfigurationError(message="missing key")
    envelope = serialize_cognee_error(exc)

    assert envelope["code"] == ErrorCode.MISSING_CONFIG.value


def test_no_data_error_add_stage():
    exc = NoDataError(stage="add")
    envelope = serialize_cognee_error(exc)

    assert envelope["code"] == ErrorCode.DATA_NOT_READY.value
    assert envelope["details"] == {"stage": "add"}
    assert "cognee.add()" in envelope["remediation"]


def test_no_data_error_cognify_stage():
    exc = NoDataError(
        message="Dataset exists but graph is empty.",
        stage="cognify",
    )
    envelope = serialize_cognee_error(exc)

    assert envelope["details"] == {"stage": "cognify"}
    assert "cognify()" in envelope["remediation"]


def test_registry_dataset_not_found():
    exc = DatasetNotFoundError()
    envelope = serialize_cognee_error(exc)

    assert envelope["code"] == ErrorCode.DATA_NOT_READY.value


def test_registry_permission_denied():
    exc = UnauthorizedDataAccessError()
    envelope = serialize_cognee_error(exc)

    assert envelope["code"] == ErrorCode.PERMISSION_DENIED.value


def test_http_error_content_keeps_legacy_detail():
    exc = NoDataError(stage="cognify")
    body = http_error_content(exc)

    assert body["detail"] == exc.message
    assert body["error"]["code"] == ErrorCode.DATA_NOT_READY.value
    assert body["error"]["retryable"] is False


def test_to_dict_on_exception():
    exc = CogneeValidationError(message="nope")
    assert exc.to_dict()["code"] == ErrorCode.INVALID_INPUT.value


def test_base_error_no_longer_uses_418():
    from fastapi import status

    from cognee.exceptions import CogneeApiError

    exc = CogneeApiError()
    assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
