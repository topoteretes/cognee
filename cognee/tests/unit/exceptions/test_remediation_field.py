"""``CogneeApiError.remediation`` reaches every transport exactly once.

The CLI already printed a first-run hint from a substring table; this checks the
hint now also rides on the exception itself (``str()``), in the REST error body,
and that the two mechanisms never double up.
"""

import json

import pytest

from cognee.exceptions import (
    REMEDIATION_MARKER,
    CogneeApiError,
    CogneeValidationError,
    find_remediation,
    remediation_for,
)
from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.tasks.ingestion.exceptions.exceptions import (
    S3FileSystemNotFoundError,
    UnsupportedDBProviderError,
)


def test_str_appends_remediation_once():
    err = CogneeApiError("boom", "Boom", remediation="set X=1")
    assert str(err) == f"Boom: boom (Status code: 500){REMEDIATION_MARKER}set X=1"
    assert str(CogneeApiError("boom", "Boom")).count("Fix:") == 0


def test_intermediate_roots_forward_remediation():
    err = CogneeValidationError("bad", remediation="pass a UUID")
    assert err.remediation == "pass a UUID"
    assert str(err).endswith("pass a UUID")


@pytest.mark.parametrize(
    "error,anchor",
    [
        (LLMAPIKeyNotSetError(), "LLM_API_KEY"),
        (S3FileSystemNotFoundError(), "cognee[aws]"),
        (UnsupportedDBProviderError(), "DB_PROVIDER"),
    ],
)
def test_first_run_errors_name_the_fix(error, anchor):
    assert anchor in error.remediation
    assert anchor in str(error)


def test_remediation_for_prefers_own_hint_and_never_repeats():
    own = CogneeApiError("boom", "Boom", remediation="set X=1")
    assert remediation_for(own) == "set X=1"

    # A cognee error re-wrapped as a plain exception: str() already carries the fix.
    wrapped = RuntimeError(str(LLMAPIKeyNotSetError()))
    assert find_remediation(str(wrapped)) is not None  # the table would match...
    assert remediation_for(wrapped) is None  # ...but the message already has it

    foreign = RuntimeError("AuthenticationError: invalid api key")
    assert remediation_for(foreign) == find_remediation(str(foreign))
    assert remediation_for(RuntimeError("unrelated")) is None


@pytest.mark.asyncio
async def test_rest_handler_adds_remediation_key_only_when_known():
    from cognee.api.client import exception_handler

    with_fix = await exception_handler(None, LLMAPIKeyNotSetError())
    body = json.loads(with_fix.body)
    assert body["detail"] == "LLM API key is not set. [LLMAPIKeyNotSetError]"
    assert "LLM_API_KEY" in body["remediation"]
    assert with_fix.status_code == 422

    without = await exception_handler(None, CogneeApiError("plain", "Plain"))
    assert set(json.loads(without.body)) == {"detail"}
