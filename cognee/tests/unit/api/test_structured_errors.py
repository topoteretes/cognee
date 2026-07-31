from cognee.exceptions import CogneeApiError, ErrorCode, http_error_content
from cognee.modules.retrieval.exceptions.exceptions import NoDataError


def test_http_error_content_structured_shape():
    exc = NoDataError(stage="cognify")
    body = http_error_content(exc)

    assert body["detail"] == exc.message
    assert body["error"]["code"] == ErrorCode.DATA_NOT_READY.value
    assert body["error"]["schema_version"] == 1
    assert "cognify()" in body["error"]["remediation"]


def test_http_error_content_system_default():
    exc = CogneeApiError(message="boom", name="TestError")
    body = http_error_content(exc)
    assert body["error"]["code"] == ErrorCode.SYSTEM.value
