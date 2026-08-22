"""Tests for the pipeline failure taxonomy (classify_pipeline_error)."""

import pytest

from cognee.modules.pipelines.operations.classify_pipeline_error import (
    classify_pipeline_error,
)


class FakeAuthenticationError(Exception):
    pass


# Classes named like the real provider/infra exceptions the rules key on.
AuthenticationError = type("AuthenticationError", (Exception,), {})
RateLimitError = type("RateLimitError", (Exception,), {})
DatabaseNotCreatedError = type("DatabaseNotCreatedError", (Exception,), {})


@pytest.mark.parametrize(
    "error,expected",
    [
        (AuthenticationError("boom"), "auth"),
        (Exception("Error code: 401 - Incorrect API key provided"), "auth"),
        (RateLimitError("slow down"), "provider_quota"),
        (Exception("You exceeded your quota: insufficient_quota"), "provider_quota"),
        (Exception("Invalid schema for response_format 'X': 'oneOf' is not permitted"), "schema"),
        (DatabaseNotCreatedError("no db"), "db_init"),
        (Exception("no such table: pipeline_runs"), "db_init"),
        (Exception("Embedding dimension mismatch for provider"), "config"),
        (Exception("unsupported vector database provider 'foo'"), "config"),
        (Exception("could not parse document with loader"), "loader"),
        (Exception("something entirely novel"), "unknown"),
        (None, "unknown"),
        ("plain rate limit text", "provider_quota"),
    ],
)
def test_classification(error, expected):
    info = classify_pipeline_error(error)
    assert info.category == expected
    assert info.remedy  # every category ships a remedy


def test_cause_chain_is_walked():
    root = AuthenticationError("Incorrect API key provided")
    try:
        try:
            raise root
        except AuthenticationError as inner:
            raise RuntimeError("pipeline step failed") from inner
    except RuntimeError as wrapped:
        info = classify_pipeline_error(wrapped)
    assert info.category == "auth"


def test_first_error_attribute_pattern():
    """The run_tasks wrapper exposes the root cause via `first_error`."""
    from cognee.modules.pipelines.exceptions import PipelineRunFailedError

    failure = PipelineRunFailedError(message="Pipeline run failed.")
    failure.first_error = AuthenticationError("Incorrect API key provided")
    root = getattr(failure, "first_error", None) or failure
    assert classify_pipeline_error(root).category == "auth"


def test_cognify_failed_error_carries_taxonomy():
    from cognee.modules.pipelines.exceptions import CognifyFailedError

    error = CognifyFailedError(
        dataset_name="main_dataset",
        error_category="auth",
        error_class="AuthenticationError",
        error_message="invalid api key",
        remedy="set LLM_API_KEY",
        duration_seconds=3.2,
    )
    text = str(error)
    assert "main_dataset" in text
    assert "[auth]" in text
    assert "AuthenticationError: invalid api key" in text
    assert "set LLM_API_KEY" in text
    assert error.error_category == "auth"


def test_raise_if_cognify_errored():
    from uuid import uuid4

    from cognee.api.v1.cognify.cognify import raise_if_cognify_errored
    from cognee.modules.pipelines.exceptions import CognifyFailedError
    from cognee.modules.pipelines.models.PipelineRunInfo import (
        PipelineRunCompleted,
        PipelineRunErrored,
    )

    common = {"pipeline_run_id": uuid4(), "dataset_id": uuid4(), "dataset_name": "ds"}

    # Completed runs pass through silently.
    raise_if_cognify_errored({"ds": PipelineRunCompleted(**common)})
    raise_if_cognify_errored(None)

    errored = PipelineRunErrored(
        **common,
        payload="AuthenticationError('invalid api key')",
        error_class="AuthenticationError",
        error_category="auth",
        error_message="invalid api key",
        remedy="set LLM_API_KEY",
        duration_seconds=1.5,
    )
    with pytest.raises(CognifyFailedError) as exc_info:
        raise_if_cognify_errored({"ds": errored})
    assert exc_info.value.error_category == "auth"
    assert exc_info.value.error_class == "AuthenticationError"
    assert exc_info.value.dataset_name == "ds"

    # A bare errored run info (no dataset mapping) raises too.
    with pytest.raises(CognifyFailedError):
        raise_if_cognify_errored(errored)


def test_wrap_cognify_exception():
    """Run-level pipeline crashes wrap into the same typed error surface."""
    from cognee.api.v1.cognify.cognify import _wrap_cognify_exception
    from cognee.modules.pipelines.exceptions import CognifyFailedError

    wrapped = _wrap_cognify_exception(AuthenticationError("invalid api key"), ["main_dataset"])
    assert isinstance(wrapped, CognifyFailedError)
    assert wrapped.error_category == "auth"
    assert wrapped.error_class == "AuthenticationError"

    # Already-typed errors pass through unchanged: no double-wrapping.
    assert _wrap_cognify_exception(wrapped, None) is wrapped
