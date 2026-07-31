import pytest

from cognee.exceptions import ErrorCode, mcp_error_payload, serialize_cognee_error
from cognee.modules.retrieval.exceptions.exceptions import NoDataError

mcp = pytest.importorskip("mcp")

MCP_SRC = __import__("pathlib").Path(__file__).resolve().parents[1] / "src"
import sys

if str(MCP_SRC) not in sys.path:
    sys.path.insert(0, str(MCP_SRC))

from error_envelope import tool_error_result  # noqa: E402


def test_tool_error_result_structured_content():
    exc = NoDataError(stage="cognify")
    result = tool_error_result(exc)

    assert result.isError is True
    assert result.structuredContent is not None
    assert result.structuredContent["error"]["code"] == ErrorCode.DATA_NOT_READY.value
    assert "cognify()" in result.structuredContent["error"]["remediation"]
    assert result.structuredContent == mcp_error_payload(exc)
