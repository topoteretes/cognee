"""Copied from cognee.modules.integrations.google.api_retry (cognee dev, 2026-11-09).

googleapiclient's retry policy, one ``execute(num_retries=0)`` per attempt through
``with_retries``, so the host's check can run between attempts.

Copied from googleapiclient 2.200.0 (``http._should_retry_response``, ``http._retry_request``).
This copy, not the installed library, decides what is retried across the declared
``>=2.100,<3`` range; compare it with the library when the pin moves. Errors are recognised
by shape (``resp.status``, ``content``), so the Google libraries are never imported here.
"""

import errno
import json
import logging
import random
import socket
import ssl
import sys
from collections.abc import Callable
from typing import TypeVar

from cognee_connectors_core.dlt_conventions import with_retries

_T = TypeVar("_T")
logger = logging.getLogger(__name__)

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
    sleep: Callable[[float], None] | None = None,
    rand: Callable[[], float] = random.random,
) -> _T:
    """Run one Google API request (``attempt`` passes ``num_retries=0``) with ``num_retries``
    retries; retry n waits ``rand() * 2**n`` seconds. ``check_active`` is as for
    ``with_retries``.
    """

    def delay(error: BaseException, number: int) -> float | None:
        if not _is_transient(error):
            return None
        wait = rand() * 2 ** (number + 1)
        status = getattr(getattr(error, "resp", None), "status", None)
        cause = f"status {status}" if isinstance(status, int) else type(error).__name__
        logger.warning(
            "Google API request failed (%s); retry %d of %d in %.1fs.",
            cause,
            number + 1,
            num_retries,
            wait,
        )
        return wait

    return with_retries(
        attempt,
        retry_on=(Exception,),
        delay=delay,
        attempts=num_retries + 1,
        check_active=host_checkpoint(check_active),
        sleep=sleep,
    )
