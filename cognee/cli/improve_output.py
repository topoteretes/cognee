"""Render an ``improve()`` answer for the terminal.

Besides the memify-shaped pipeline-run mapping, ``improve(session_ids=...)`` can
answer with a status object (SDK-593): ``no_op`` (nothing above the session
watermarks), ``busy`` (another improve of that session is in flight and will cover
the newer tail) or ``accepted`` (background run started). Both the local command
and the ``--api-url`` dispatch print those the same way.
"""

from typing import Any

import cognee.cli.echo as fmt


def improve_status(result: Any) -> str | None:
    """The status token of a status-shaped improve answer, else None."""
    if isinstance(result, dict) and isinstance(result.get("status"), str):
        return result["status"]
    return None


def echo_improve_result(result: Any, *, background: bool) -> None:
    status = improve_status(result)
    if status is None:
        if background:
            fmt.success("Improvement started in background!")
        else:
            fmt.success("Knowledge graph improved successfully!")
        if isinstance(result, dict):
            for ds_id, run_info in result.items():
                run_status = getattr(run_info, "status", None)
                if run_status is None and isinstance(run_info, dict):
                    run_status = run_info.get("status", run_info)
                fmt.echo(f"  Dataset {ds_id}: {run_status if run_status is not None else run_info}")
        return

    sessions = ", ".join(result.get("session_ids") or []) or "-"
    if status == "no_op":
        reason = result.get("reason") or "nothing_pending"
        fmt.success(f"Nothing to improve for session(s) {sessions} ({reason}).")
    elif status == "busy":
        age = result.get("holder_age_seconds")
        age_text = f" for {int(age)} s" if isinstance(age, (int, float)) else ""
        fmt.warning(
            f"Another improve of session {result.get('session_id') or sessions} is already "
            f"running{age_text}; it will cover the newer entries before it finishes. "
            "No retry needed."
        )
    elif status == "accepted":
        stages = ", ".join(result.get("pending_stages") or []) or "enrichment"
        fmt.success(f"Improvement accepted and running in background (stages: {stages}).")
    else:
        fmt.echo(f"Improve status: {status}")
