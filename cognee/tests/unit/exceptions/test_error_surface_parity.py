import json

import pytest

from cognee.cli.error_emit import emit_cognee_error_json
from cognee.exceptions import (
    CogneeConfigurationError,
    CogneeTransientError,
    http_error_content,
    mcp_error_payload,
    serialize_cognee_error,
)
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.modules.data.exceptions.exceptions import UnauthorizedDataAccessError
from cognee.modules.retrieval.exceptions.exceptions import NoDataError

REPRESENTATIVE_EXCEPTIONS = [
    NoDataError(stage="cognify"),
    CogneeConfigurationError("missing key"),
    CogneeTransientError(),
    UnauthorizedDataAccessError(),
    LLMAPIKeyNotSetError(),
]


@pytest.mark.parametrize("exc", REPRESENTATIVE_EXCEPTIONS)
def test_http_mcp_same_error_object(exc):
    http_error = http_error_content(exc)["error"]
    mcp_error = mcp_error_payload(exc)["error"]
    assert http_error == mcp_error


@pytest.mark.parametrize("exc", REPRESENTATIVE_EXCEPTIONS)
def test_cli_json_matches_serializer(exc, capsys):
    emit_cognee_error_json(exc)
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["error"] == serialize_cognee_error(exc)
