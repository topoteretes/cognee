"""What an error event says about its error (SDK-775): class names and a status.

``telemetry_exception_properties`` follows the ``raise ... from`` chain so the
provider error under a cognee wrapper is visible, and carries the HTTP status
closest to the failure. Never a message.
"""

from itertools import pairwise

import pytest

from cognee.shared import utils

SECRET_MESSAGE = "quota exceeded for dataset 'customer-secrets-2026'"


class ProviderError(Exception):
    status_code = 429


class WrapperError(Exception):
    pass


def _raise_wrapped():
    try:
        raise ProviderError(SECRET_MESSAGE)
    except ProviderError as error:
        raise WrapperError("wrapped") from error


def _caught(raiser):
    with pytest.raises(BaseException) as info:
        raiser()
    return info.value


def test_bare_error_reports_its_type_only():
    properties = utils.telemetry_exception_properties(ValueError(SECRET_MESSAGE))

    assert properties == {"exception_type": "ValueError"}
    assert SECRET_MESSAGE not in repr(properties)


def test_cause_chain_and_provider_status_travel_with_the_wrapper():
    properties = utils.telemetry_exception_properties(_caught(_raise_wrapped))

    assert properties == {
        "exception_type": "WrapperError",
        "exception_chain": ["WrapperError", "ProviderError"],
        "exception_cause": "ProviderError",
        "status_code": 429,
    }
    assert SECRET_MESSAGE not in repr(properties)


def test_context_is_followed_unless_suppressed():
    def raise_in_handler():
        try:
            raise KeyError("k")
        except KeyError:
            raise RuntimeError("while handling")

    def raise_from_none():
        try:
            raise KeyError("k")
        except KeyError:
            raise RuntimeError("explicitly detached") from None

    assert (
        utils.telemetry_exception_properties(_caught(raise_in_handler))["exception_cause"]
        == "KeyError"
    )
    assert utils.telemetry_exception_properties(_caught(raise_from_none)) == {
        "exception_type": "RuntimeError"
    }


def test_cause_is_omitted_when_it_repeats_the_type():
    def raise_same():
        try:
            raise ValueError("inner")
        except ValueError as error:
            raise ValueError("outer") from error

    properties = utils.telemetry_exception_properties(_caught(raise_same))

    assert properties["exception_chain"] == ["ValueError", "ValueError"]
    assert "exception_cause" not in properties


def test_chain_is_bounded_and_cycle_safe():
    errors = [RuntimeError(str(depth)) for depth in range(8)]
    for outer, inner in pairwise(errors):
        outer.__cause__ = inner
    deep = utils.telemetry_exception_properties(errors[0])
    assert len(deep["exception_chain"]) == utils.TELEMETRY_EXCEPTION_CHAIN_DEPTH

    loop = RuntimeError("loop")
    loop.__cause__ = loop
    assert utils.telemetry_exception_properties(loop) == {"exception_type": "RuntimeError"}


def test_first_error_is_unwrapped_before_the_chain_is_read():
    """A ``PipelineRunFailedError`` reports the item error it wraps, chain included."""
    run_error = RuntimeError("Pipeline run failed.")
    run_error.first_error = _caught(_raise_wrapped)

    properties = utils.telemetry_exception_properties(run_error)

    assert properties["exception_type"] == "WrapperError"
    assert properties["exception_cause"] == "ProviderError"
    assert properties["status_code"] == 429
    assert utils.telemetry_exception_type(run_error) == "WrapperError"


@pytest.mark.parametrize(
    "status_code, expected",
    [
        (503, 503),
        ("401", 401),
        (True, None),  # a bool is not a status
        ("rate limited", None),
        (42, None),  # outside 100-599
        (None, None),
    ],
)
def test_only_an_integer_http_status_is_reported(status_code, expected):
    error = RuntimeError("x")
    error.status_code = status_code

    properties = utils.telemetry_exception_properties(error)

    assert properties.get("status_code") == expected


def test_the_innermost_status_wins():
    """The provider's status, not the cognee error's mapped one, says why."""

    def raise_mapped():
        try:
            raise ProviderError("429 from the provider")
        except ProviderError as error:
            mapped = WrapperError("mapped")
            mapped.status_code = 500
            raise mapped from error

    assert utils.telemetry_exception_properties(_caught(raise_mapped))["status_code"] == 429
