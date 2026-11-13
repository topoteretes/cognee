from collections.abc import Callable, Iterable
from typing import Any


def complete_id_snapshot(pages: Iterable[Iterable[dict[str, Any]]]) -> set[str]:
    """Collect IDs and fail if any page is incomplete."""
    found: set[str] = set()
    for page in pages:
        if page is None:
            raise RuntimeError("Meta Ads ID listing returned an incomplete page")
        for item in page:
            if not item.get("id"):
                raise RuntimeError("Meta Ads ID listing contained a row without an ID")
            found.add(str(item["id"]))
    return found


def reconcile_ids(
    stored_ids: set[str],
    pages: Iterable[Iterable[dict[str, Any]]],
    delete: Callable[[set[str]], Any],
) -> set[str]:
    """Delete IDs only after a complete listing succeeds."""
    try:
        current_ids = complete_id_snapshot(pages)
    except Exception:
        return set()
    removed = stored_ids - current_ids
    if removed:
        delete(removed)
    return removed
