"""googleapiclient's retry policy, run through the shared ``with_retries``.

``request.execute(num_retries=N)`` retries inside the library, where the host's
authorization callback cannot run: a revoked connection kept retrying and sleeping. Here
each attempt is one ``request.execute(num_retries=0)`` and ``with_retries`` repeats it, so
the callback runs before every attempt and every backoff wait. Which answers are retried,
how many times and how long the wait is stay the library's own (googleapiclient 2.200.0,
``http._should_retry_response`` and ``http._retry_request``; re-check on an upgrade).

The Google libraries are not imported: a pre-made service never needs them, so errors are
recognised by shape (``resp.status`` and ``content`` of an ``HttpError``), which is also why
``retry_on`` is ``Exception`` and ``_retry_delay`` is the filter.
"""

import errno
import json
import random
import socket
import ssl
import sys
import time
from collections.abc import Callable
from typing import TypeVar

from cognee.tasks.ingestion.dlt_utils import with_retries

_T = TypeVar("_T")

# http._retry_request retries a bare OSError only for these errno names; names, not numbers,
# because the numbers differ by platform.
_RETRIED_ERRNO_NAMES = frozenset(
    {"WSAETIMEDOUT", "ETIMEDOUT", "EPIPE", "ECONNABORTED", "ECONNREFUSED", "ECONNRESET"}
)
# http._should_retry_response retries a 403 only for these reasons.
_RATE_LIMIT_REASONS = ("userRateLimitExceeded", "rateLimitExceeded")


# Set on what a host checkpoint raised (see ``host_checkpoint``).
_HOST_STOP_ATTR = "_cognee_host_stop"


def host_checkpoint(check_active: Callable[[], None] | None) -> Callable[[], None] | None:
    """``check_active`` marking what it raises, so ``is_host_stop`` can tell it from a Google error.

    A connector's blanket ``except Exception`` around a request (one that wraps the error,
    counts it as a failed fetch or swallows it) would otherwise treat a revoked connection
    like a failed request. The error itself is raised unchanged.
    """
    if check_active is None:
        return None

    def checkpoint() -> None:
        try:
            check_active()
        except Exception as error:
            setattr(error, _HOST_STOP_ATTR, True)
            raise

    return checkpoint


def is_host_stop(error: BaseException) -> bool:
    """Whether ``error`` was raised by a ``host_checkpoint``; such an error is always re-raised."""
    return getattr(error, _HOST_STOP_ATTR, False)


def _sleep(seconds: float) -> None:
    # Looked up at call time, so a test can replace ``time.sleep`` without a parameter.
    time.sleep(seconds)


def _should_retry_response(status: int, content: bytes | None) -> bool:
    """Whether Google's answer is transient: 5xx, 429, or a 403 for a rate-limit reason.

    A 403 with no body is not retried. Like the library, a body that is not JSON is not
    retried either.
    """
    if status >= 500 or status == 429:
        return True
    if status != 403 or not content:
        return False
    try:
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, dict):
            # "errors" first, as the library reads it (googleapiclient issue 1243).
            keyword = next(
                (kw for kw in ("errors", "status", "message") if kw in data["error"]), ""
            )
            reason = data["error"][keyword] if keyword else None
            if isinstance(reason, list) and len(reason) > 0:
                reason = reason[0]
                if "reason" in reason:
                    reason = reason["reason"]
        else:
            reason = data[0]["error"]["errors"]["reason"]
    except (UnicodeDecodeError, ValueError, KeyError):
        return False
    return reason in _RATE_LIMIT_REASONS


def _is_transient(error: BaseException) -> bool:
    """Whether the library's retry loop would have retried ``error``.

    Connection errors, timeouts and TLS errors, a bare ``OSError`` with one of the errno
    names above, httplib2's ``ServerNotFoundError``, and an ``HttpError`` whose answer
    ``_should_retry_response`` accepts.
    """
    if isinstance(error, (ssl.SSLError, socket.timeout, ConnectionError)):
        return True
    if isinstance(error, OSError):
        return errno.errorcode.get(error.errno) in _RETRIED_ERRNO_NAMES
    # An httplib2 error can only be raised once httplib2 is imported, so it needs no import.
    httplib2 = sys.modules.get("httplib2")
    if httplib2 is not None and isinstance(error, httplib2.ServerNotFoundError):
        return True
    status = getattr(getattr(error, "resp", None), "status", None)
    if isinstance(status, int):
        return _should_retry_response(status, getattr(error, "content", None))
    return False


def with_google_retries(
    attempt: Callable[[], _T],
    *,
    num_retries: int,
    check_active: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = _sleep,
    rand: Callable[[], float] = random.random,
) -> _T:
    """Run one Google API request with ``num_retries`` retries, as ``num_retries=`` did.

    ``attempt`` makes exactly one request and must pass ``num_retries=0`` to the library
    (``lambda: request.execute(num_retries=0)``): ``num_retries`` N means N + 1 attempts, and
    retry n waits ``rand() * 2**n`` seconds. ``check_active`` runs before every attempt and
    every wait, outside the retried call, and what it raises propagates as it is.
    """

    def delay(error: BaseException, number: int) -> float | None:
        return rand() * 2 ** (number + 1) if _is_transient(error) else None

    return with_retries(
        attempt,
        retry_on=(Exception,),
        delay=delay,
        attempts=num_retries + 1,
        check_active=host_checkpoint(check_active),
        sleep=sleep,
    )
