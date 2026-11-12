"""Read surface of the audit-grade provenance ledger.

Every endpoint is read-only — the ledger is written by ingestion and by the
delete paths (tombstones), never through HTTP. Access control follows the
dataset namespace baked into every ledger key
(``{dataset_id}:data_item:{data_id}``): a per-entity lookup is allowed when the caller can
read the dataset the key belongs to, and the ledger-wide walks (verify,
statistics, export) require a ``dataset_id`` unless the caller is a
superuser. Keys without a dataset prefix (custom pipelines that bypass the
writer's namespacing) are superuser-only for the same reason. The rule itself
lives in ``cognee.modules.provenance.access`` and is shared with the CLI.

The ledger is off by default (``PROVENANCE_TRACKING``). While it is off every
route but ``/status`` answers 409 with that reason, instead of reading a
ledger nobody is writing: "valid, 0 rows" would look like a verified ledger.
"""

import json
from collections.abc import Awaitable
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, StreamingResponse

from cognee.modules.provenance import get_provenance_manager
from cognee.modules.provenance.access import (
    authorize_anchoring,
    authorize_ledger_key,
    authorize_ledger_scope,
)
from cognee.modules.provenance.tracking import (
    ProvenanceTrackingDisabledError,
    provenance_tracking_enabled,
    require_provenance_tracking,
)
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage

logger = get_logger()


async def _authorize(check: Awaitable[None]) -> None:
    """Run one ledger access check (``cognee.modules.provenance.access``); a denial is a 403."""
    try:
        await check
    except PermissionDeniedError as error:
        raise HTTPException(status_code=403, detail=error.message) from error


def _require_ledger_on() -> None:
    """Answer 409 with the reason while the ledger is off (see the module docstring)."""
    try:
        require_provenance_tracking()
    except ProvenanceTrackingDisabledError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def get_provenance_router() -> APIRouter:
    router = APIRouter(
        responses={
            409: {
                "description": "The audit ledger is off: set `PROVENANCE_TRACKING=true` "
                "(it is off by default). On the anchor routes, also returned when "
                "anchoring is not configured."
            }
        }
    )

    @router.get("/entry/{entity_id:path}", summary="Current ledger row of one entity")
    @log_usage(function_name="GET /v1/provenance/entry", log_type="api_endpoint")
    async def get_entry(
        entity_id: str = PathParam(
            ..., description="Ledger key, e.g. `{dataset_id}:data_item:{data_id}`"
        ),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """The live row (or current tombstone) for a ledger key; 404 if never tracked."""
        _require_ledger_on()
        await _authorize(authorize_ledger_key(user, entity_id))
        entry = await get_provenance_manager().get_provenance(entity_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Entity is not in the ledger.")
        return entry

    @router.get("/history/{entity_id:path}", summary="Version history of one entity")
    @log_usage(function_name="GET /v1/provenance/history", log_type="api_endpoint")
    async def get_history(
        entity_id: str = PathParam(..., description="Ledger key"),
        user: User = Depends(get_authenticated_user),
    ) -> list[dict[str, Any]]:
        """Oldest-first versions; tombstone versions carry `invalidated_by` /
        `invalidation_reason`, so a row reads "asserted by A, retracted by B"."""
        _require_ledger_on()
        await _authorize(authorize_ledger_key(user, entity_id))
        history = await get_provenance_manager().revision_history(entity_id)
        if not history:
            raise HTTPException(status_code=404, detail="Entity is not in the ledger.")
        return history

    @router.get("/verify", summary="Verify checksums and the hash chain")
    @log_usage(function_name="GET /v1/provenance/verify", log_type="api_endpoint")
    async def verify(
        dataset_id: UUID | None = Query(default=None),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """Tamper evidence. Without `dataset_id` (superusers) the whole chain is
        walked; with it, each of the dataset's rows is checked against the
        checksum stored at the previous chain position."""
        _require_ledger_on()
        await _authorize(authorize_ledger_scope(user, dataset_id))
        return await get_provenance_manager().verify_chain(dataset_id=dataset_id)

    @router.get("/statistics", summary="Ledger statistics")
    @log_usage(function_name="GET /v1/provenance/statistics", log_type="api_endpoint")
    async def statistics(
        dataset_id: UUID | None = Query(default=None),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """Row counts (total / live / archived), per-type counts, distinct sources,
        and live tombstones."""
        _require_ledger_on()
        await _authorize(authorize_ledger_scope(user, dataset_id))
        return await get_provenance_manager().get_statistics(dataset_id=dataset_id)

    @router.get("/export", summary="Export ledger rows as JSON Lines")
    @log_usage(function_name="GET /v1/provenance/export", log_type="api_endpoint")
    async def export(
        dataset_id: UUID | None = Query(default=None),
        include_archived: bool = Query(
            default=True, description="Include archived versions (needed to re-verify history)"
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """Streams one JSON object per line in chain order, so the export can be
        re-verified offline with `cognee.modules.provenance.integrity`."""
        _require_ledger_on()
        await _authorize(authorize_ledger_scope(user, dataset_id))
        manager = get_provenance_manager()

        async def lines():
            async for row in manager.export(
                dataset_id=dataset_id, include_archived=include_archived
            ):
                yield json.dumps(row, default=str) + "\n"

        suffix = f"-{dataset_id}" if dataset_id is not None else ""
        return StreamingResponse(
            lines(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="provenance{suffix}.jsonl"'},
        )

    @router.post("/anchor", summary="Sign and record the current chain head")
    @log_usage(function_name="POST /v1/provenance/anchor", log_type="api_endpoint")
    async def anchor(user: User = Depends(get_authenticated_user)) -> dict[str, Any]:
        """Superuser only. Appends an HMAC-signed `(sequence_id, checksum)` anchor
        to the external anchor file (`PROVENANCE_ANCHOR_KEY` / `PROVENANCE_ANCHOR_PATH`).
        Returns the anchor; `{"anchor": null}` on an empty ledger."""
        _require_ledger_on()
        from cognee.modules.provenance.anchors import AnchoringNotConfiguredError

        await _authorize(authorize_anchoring(user))
        try:
            return {"anchor": await get_provenance_manager().anchor()}
        except AnchoringNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/anchors/verify", summary="Verify the ledger against its external anchors")
    @log_usage(function_name="GET /v1/provenance/anchors/verify", log_type="api_endpoint")
    async def verify_anchors(user: User = Depends(get_authenticated_user)) -> dict[str, Any]:
        """Superuser only. Each anchor's signature must verify and the row at its
        chain position must still carry the anchored checksum."""
        _require_ledger_on()
        from cognee.modules.provenance.anchors import AnchoringNotConfiguredError

        await _authorize(authorize_anchoring(user))
        try:
            return await get_provenance_manager().verify_anchors()
        except AnchoringNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/status", summary="Whether the audit ledger is being written")
    async def status(user: User = Depends(get_authenticated_user)) -> JSONResponse:
        """`PROVENANCE_TRACKING` as the server sees it. The ledger is off by
        default; while it is off the other routes answer 409 and nothing is
        recorded. Rows recorded earlier are served again once it is back on."""
        from cognee.modules.provenance.anchors import anchoring_configured

        return JSONResponse(
            content={
                "provenance_tracking": provenance_tracking_enabled(),
                "anchoring_configured": anchoring_configured(),
            }
        )

    return router
