"""Read surface of the audit-grade provenance ledger.

Every endpoint is read-only — the ledger is written by ingestion and by the
delete paths (tombstones), never through HTTP. Access control follows the
dataset namespace baked into every ledger key (``{dataset_id}:{node_id}`` /
``rel:{dataset_id}:...``): a per-entity lookup is allowed when the caller can
read the dataset the key belongs to, and the ledger-wide walks (verify,
check, statistics, export) require a ``dataset_id`` unless the caller is a
superuser. Keys without a dataset prefix (custom pipelines that bypass the
writer's namespacing) are superuser-only for the same reason.
"""

import json
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, StreamingResponse

from cognee.modules.data.methods import get_authorized_dataset
from cognee.modules.provenance import get_provenance_manager
from cognee.modules.provenance.tombstones import dataset_id_from_ledger_key
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage

logger = get_logger()


async def _authorize_dataset(user: User, dataset_id: UUID) -> None:
    if await get_authorized_dataset(user, dataset_id, "read") is None:
        raise HTTPException(status_code=403, detail=f"No read access to dataset {dataset_id}.")


async def _authorize_scope(user: User, dataset_id: UUID | None) -> None:
    """Ledger-wide reads are superuser-only; scoped reads need dataset read access."""
    if dataset_id is None:
        if not getattr(user, "is_superuser", False):
            raise HTTPException(
                status_code=403,
                detail="dataset_id is required; the whole ledger is readable by superusers only.",
            )
        return
    await _authorize_dataset(user, dataset_id)


async def _authorize_key(user: User, entity_id: str) -> None:
    dataset_id = dataset_id_from_ledger_key(entity_id)
    if dataset_id is None:
        if not getattr(user, "is_superuser", False):
            raise HTTPException(
                status_code=403, detail="Unscoped ledger keys are readable by superusers only."
            )
        return
    await _authorize_dataset(user, dataset_id)


def get_provenance_router() -> APIRouter:
    router = APIRouter()

    @router.get("/entry/{entity_id:path}", summary="Current ledger row of one entity")
    @log_usage(function_name="GET /v1/provenance/entry", log_type="api_endpoint")
    async def get_entry(
        entity_id: str = PathParam(..., description="Ledger key, e.g. `{dataset_id}:{node_id}`"),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """The live row (or current tombstone) for a ledger key; 404 if never tracked."""
        await _authorize_key(user, entity_id)
        entry = await get_provenance_manager().get_provenance(entity_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Entity is not in the ledger.")
        return entry

    @router.get("/lineage/{entity_id:path}", summary="Upstream lineage of one entity")
    @log_usage(function_name="GET /v1/provenance/lineage", log_type="api_endpoint")
    async def get_lineage(
        entity_id: str = PathParam(..., description="Ledger key"),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """Everything the entity was derived from (parent + used entities, BFS),
        with per-row checksums re-verified (`integrity_verified`)."""
        await _authorize_key(user, entity_id)
        lineage = await get_provenance_manager().get_lineage(entity_id)
        if not lineage:
            raise HTTPException(status_code=404, detail="Entity is not in the ledger.")
        return lineage

    @router.get("/history/{entity_id:path}", summary="Version history of one entity")
    @log_usage(function_name="GET /v1/provenance/history", log_type="api_endpoint")
    async def get_history(
        entity_id: str = PathParam(..., description="Ledger key"),
        user: User = Depends(get_authenticated_user),
    ) -> list[dict[str, Any]]:
        """Oldest-first versions; tombstone versions carry `invalidated_by` /
        `invalidation_reason`, so a row reads "asserted by A, retracted by B"."""
        await _authorize_key(user, entity_id)
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
        await _authorize_scope(user, dataset_id)
        return await get_provenance_manager().verify_chain(dataset_id=dataset_id)

    @router.get("/check", summary="Referential-integrity check")
    @log_usage(function_name="GET /v1/provenance/check", log_type="api_endpoint")
    async def check(
        dataset_id: UUID | None = Query(default=None),
        strict: bool = Query(default=False),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """Dangling lineage links (parent / used / version / activity references)."""
        await _authorize_scope(user, dataset_id)
        return await get_provenance_manager().check(strict=strict, dataset_id=dataset_id)

    @router.get("/drift", summary="Compare the ledger's snapshots with the graph")
    @log_usage(function_name="GET /v1/provenance/drift", log_type="api_endpoint")
    async def drift(
        dataset_id: UUID = Query(...),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """Each live node row carries a content snapshot; this re-reads the node
        from the graph and reports rows whose content changed out of band
        (`drifted`, with a field delta) or vanished without a tombstone
        (`missing_in_graph`)."""
        dataset = await get_authorized_dataset(user, dataset_id, "read")
        if dataset is None:
            raise HTTPException(status_code=403, detail=f"No read access to dataset {dataset_id}.")
        return await get_provenance_manager().check_drift(dataset_id, dataset.owner_id)

    @router.get("/statistics", summary="Ledger statistics")
    @log_usage(function_name="GET /v1/provenance/statistics", log_type="api_endpoint")
    async def statistics(
        dataset_id: UUID | None = Query(default=None),
        user: User = Depends(get_authenticated_user),
    ) -> dict[str, Any]:
        """Row counts (total / live / archived), per-type counts, distinct sources,
        and live tombstones."""
        await _authorize_scope(user, dataset_id)
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
        await _authorize_scope(user, dataset_id)
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
        from cognee.modules.provenance.anchors import AnchoringNotConfiguredError

        if not getattr(user, "is_superuser", False):
            raise HTTPException(status_code=403, detail="Anchoring is superuser-only.")
        try:
            return {"anchor": await get_provenance_manager().anchor()}
        except AnchoringNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/anchors/verify", summary="Verify the ledger against its external anchors")
    @log_usage(function_name="GET /v1/provenance/anchors/verify", log_type="api_endpoint")
    async def verify_anchors(user: User = Depends(get_authenticated_user)) -> dict[str, Any]:
        """Superuser only. Each anchor's signature must verify and the row at its
        chain position must still carry the anchored checksum."""
        from cognee.modules.provenance.anchors import AnchoringNotConfiguredError

        if not getattr(user, "is_superuser", False):
            raise HTTPException(status_code=403, detail="Anchor verification is superuser-only.")
        try:
            return await get_provenance_manager().verify_anchors()
        except AnchoringNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/status", summary="Whether the audit ledger is being written")
    async def status(user: User = Depends(get_authenticated_user)) -> JSONResponse:
        """`PROVENANCE_TRACKING` as the server sees it. Reads work either way —
        a ledger written earlier stays readable after the flag is turned off."""
        from cognee.modules.provenance.anchors import anchoring_configured
        from cognee.tasks.provenance.record_provenance import provenance_tracking_enabled

        return JSONResponse(
            content={
                "provenance_tracking": provenance_tracking_enabled(),
                "anchoring_configured": anchoring_configured(),
            }
        )

    return router
