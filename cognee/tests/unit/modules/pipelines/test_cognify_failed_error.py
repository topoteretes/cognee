"""Tests for the loud-first-build failure surface (CognifyFailedError)."""

import pytest

AuthenticationError = type("AuthenticationError", (Exception,), {})


def test_first_error_attribute_pattern():
    """The run_tasks wrapper exposes the root cause via `first_error`."""
    from cognee.modules.pipelines.exceptions import PipelineRunFailedError

    failure = PipelineRunFailedError(message="Pipeline run failed.")
    failure.first_error = AuthenticationError("Incorrect API key provided")
    root = getattr(failure, "first_error", None) or failure
    assert type(root).__name__ == "AuthenticationError"
    assert "Incorrect API key" in str(root)


def test_cognify_failed_error_carries_root_cause():
    from cognee.modules.pipelines.exceptions import CognifyFailedError

    error = CognifyFailedError(
        dataset_name="main_dataset",
        error_class="AuthenticationError",
        error_message="invalid api key",
    )
    text = str(error)
    assert "main_dataset" in text
    assert "AuthenticationError: invalid api key" in text
    assert "raise_on_error=False" in text
    assert error.error_class == "AuthenticationError"


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
        error_message="invalid api key",
    )
    with pytest.raises(CognifyFailedError) as exc_info:
        raise_if_cognify_errored({"ds": errored})
    assert exc_info.value.error_class == "AuthenticationError"
    assert exc_info.value.dataset_name == "ds"

    # A bare errored run info (no dataset mapping) raises too.
    with pytest.raises(CognifyFailedError):
        raise_if_cognify_errored(errored)

    # Legacy run infos without error fields still raise, quoting the payload.
    legacy = PipelineRunErrored(**common, payload="SomeError('boom')")
    with pytest.raises(CognifyFailedError) as exc_info:
        raise_if_cognify_errored({"ds": legacy})
    assert "boom" in str(exc_info.value)


def test_wrap_cognify_exception():
    """Run-level pipeline crashes wrap into the same typed error surface."""
    from cognee.api.v1.cognify.cognify import _wrap_cognify_exception
    from cognee.modules.pipelines.exceptions import CognifyFailedError

    wrapped = _wrap_cognify_exception(AuthenticationError("invalid api key"), ["main_dataset"])
    assert isinstance(wrapped, CognifyFailedError)
    assert wrapped.error_class == "AuthenticationError"
    assert "invalid api key" in str(wrapped)
    # On the run-level path raise_on_error=False re-raises the original
    # exception — the hint must not promise errored run info.
    assert "original exception" in str(wrapped)
    assert "errored run info" not in str(wrapped)

    # Already-typed errors pass through unchanged: no double-wrapping.
    assert _wrap_cognify_exception(wrapped, None) is wrapped


def test_get_errored_run_info():
    """Callers that opt out of raising use this to detect a failed build."""
    from uuid import uuid4

    from cognee.modules.pipelines.models.PipelineRunInfo import (
        PipelineRunCompleted,
        PipelineRunErrored,
        get_errored_run_info,
    )

    common = {"pipeline_run_id": uuid4(), "dataset_id": uuid4(), "dataset_name": "ds"}
    completed = PipelineRunCompleted(**common)
    errored = PipelineRunErrored(**common, error_class="AuthenticationError")

    assert get_errored_run_info(None) is None
    assert get_errored_run_info({"ds": completed}) is None
    assert get_errored_run_info({"a": completed, "b": errored}) is errored
    assert get_errored_run_info(errored) is errored
