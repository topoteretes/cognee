"""Host admission for automatic improves.

``remember()`` starts an ``improve()`` on its own: the background session
bridge after a session write, and the enrichment pass after a permanent
add + cognify. An application that embeds cognee sometimes knows before any
work starts that such a run cannot succeed — the tenant it serves has no LLM
budget left, say. With no way to say so, the run starts anyway and fails at
its first LLM call, on every ``remember()``.

This module is where the host says so. It registers ONE async check, and
``remember()`` awaits it when it decides whether an automatic improve will
run: up front on both paths, before the session debounce is consulted. A
returned reason skips the improve and is reported as
``RememberResult.improve_skipped``. The data is stored either way, and an
explicit ``improve()`` call is never gated.

The check can never break ``remember()``: nothing registered, a check that
raises, and an answer that is neither ``None`` nor a reason string all mean
"allow".
"""

from collections.abc import Awaitable, Callable
from typing import Any

from cognee.shared.logging_utils import get_logger

logger = get_logger("improve")

# ``async def check(*, user, dataset_id, session_id, **kwargs) -> str | None``
AutoImproveAdmission = Callable[..., Awaitable[str | None]]

_admission_check: AutoImproveAdmission | None = None


def register_auto_improve_admission(check: AutoImproveAdmission | None) -> None:
    """Register the check ``remember()`` consults before an automatic improve.

    One check per process: a later registration replaces the earlier one, and
    ``None`` removes it, like ``clear_auto_improve_admission()``.

    ``check`` is awaited with keyword arguments describing the automatic
    improve ``remember()`` would run, and returns ``None`` to allow it or a
    short machine-readable reason (``"insufficient_credits"``) to skip it::

        async def check(*, user, dataset_id, session_id, **kwargs) -> str | None:
            return None if await has_credit(user) else "insufficient_credits"

        register_auto_improve_admission(check)

    Keyword arguments passed today — accept ``**kwargs``, more may follow:

    * ``user``: the user the improve would run as.
    * ``dataset_id``: id of the dataset it would improve.
    * ``session_id``: the session that was written, for the session bridge;
      ``None`` for the improve that follows a permanent remember.
    * ``session_ids``: the session ids the improve would bridge into the
      graph (empty when none).
    """
    global _admission_check
    if check is not None and not callable(check):
        raise TypeError(
            "register_auto_improve_admission() takes an async callable or None, "
            f"got {type(check).__name__}"
        )
    _admission_check = check


def clear_auto_improve_admission() -> None:
    """Remove the registered check; every automatic improve is allowed again."""
    register_auto_improve_admission(None)


async def auto_improve_skip_reason(**context: Any) -> str | None:
    """Ask the registered check whether an automatic improve may start.

    Returns the check's skip reason, or ``None`` when the improve may run:
    nothing is registered, the check allowed it, or the check could not give
    a usable answer. The last case is deliberate — the check exists to save
    doomed work, so a broken one costs at most the run it would have saved,
    and never the ``remember()`` that asked.
    """
    check = _admission_check
    if check is None:
        return None

    try:
        reason = await check(**context)
    except Exception as error:
        logger.warning(
            "improve: auto-improve admission check failed, allowing the improve: %s",
            error,
            exc_info=True,
        )
        return None

    if reason is None:
        return None
    if isinstance(reason, str) and reason.strip():
        return reason

    logger.warning(
        "improve: auto-improve admission check returned %s instead of a reason string "
        "or None, allowing the improve",
        type(reason).__name__ if not isinstance(reason, str) else "an empty string",
    )
    return None
