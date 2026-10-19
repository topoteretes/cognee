"""Route explicit source questions and execute native, authorized retrieval.

Plugins delegate here rather than mapping provider names to search types. SQL
uses the existing connection registry and read-only executor; document hits
are checked against native document membership before returning evidence.
"""

import asyncio
from contextvars import Context
from datetime import datetime, timezone
from uuid import UUID

from fastapi.encoders import jsonable_encoder

from cognee.api.v1.search import search as native_search
from cognee.modules.data.source_catalog import route_sources, source_document
from cognee.modules.search.types import SearchType
from cognee.modules.tools.errors import ToolConnectionNotFoundError
from cognee.modules.tools.text_to_sql import run_text_to_sql
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Each selected source gets its own bound and runs concurrently with the others,
# so one slow source degrades only itself. The overall bound covers routing plus
# the slowest source, never the sum of the sources.
PER_SOURCE_TIMEOUT_SECONDS = 90
OVERALL_TIMEOUT_SECONDS = 270


async def _document_evidence(user, target, query, top_k):
    results = jsonable_encoder(
        await native_search(
            query_text=query,
            query_type=SearchType.CHUNKS,
            user=user,
            dataset_ids=[UUID(target["dataset_id"])],
            node_name=target["node_sets"] or None,
            node_name_filter_operator="AND",
            top_k=top_k,
            only_context=True,
            verbose=True,
        )
    )
    evidence, rejected, metadata = [], 0, {}
    for result in results:
        for hit in result.get("objects_result") or []:
            payload = hit.get("payload") or {}
            document = payload.get("document_id")
            if not document:
                rejected += 1
                continue
            if document not in metadata:
                metadata[document] = await source_document(
                    user, UUID(target["dataset_id"]), UUID(document)
                )
            row = metadata[document]
            if not set(target["node_sets"]).issubset(row.get("node_sets") or []):
                # source_document already authorized the dataset. A chunk whose
                # graph node-set tags disagree with the current Data row (re-adding
                # content rewrites Data.node_set while earlier chunks keep their
                # tags) is a consistency problem, not an access one: drop the hit
                # as unverifiable and keep searching.
                logger.warning(
                    "Rejected unverifiable hit for source %s: node sets do not match",
                    target["id"],
                )
                rejected += 1
                continue
            external = row.get("external_metadata") or {}
            evidence.append(
                {
                    "retrieval_method": "chunks",
                    "source_target": target["name"],
                    "document_id": row["id"],
                    "dataset_id": row["dataset_id"],
                    "label": row.get("label"),
                    "node_sets": row.get("node_sets"),
                    "source_url": external.get("source_url"),
                    "text": payload.get("text"),
                    "score": hit.get("score"),
                }
            )
    return evidence, rejected


async def _retrieve_target(user, target, query, top_k):
    """Retrieve one selected source: ``(evidence, error, rejected, searched)``.

    Only an authorization failure escapes, as ``PermissionError``; every other
    failure, including the per-source timeout, is returned as a sanitized error
    entry so the other sources' evidence survives.
    """
    method = target.get("retrieval_method", "chunks")
    failure = {"source_id": target["id"], "retrieval_method": method}
    try:
        if method == "sql":
            # Re-resolves the same caller's connection at execution. Native
            # guards enforce feature gate, read-only SQL, table allowlists,
            # statement timeout and row limits.
            result = await asyncio.wait_for(
                run_text_to_sql(user.id, target["connection"], query),
                timeout=PER_SOURCE_TIMEOUT_SECONDS,
            )
            if not result.success:
                return [], {**failure, "error": "Native database retrieval failed."}, 0, False
            evidence = {
                "retrieval_method": "sql",
                "source_target": target["name"],
                "queried_at": datetime.now(timezone.utc).isoformat(),
                "success": True,
                "text": result.render_text(),
                "structured": result.structured(),
            }
            return [evidence], None, 0, True
        if method == "chunks":
            hits, rejected = await asyncio.wait_for(
                _document_evidence(user, target, query, top_k),
                timeout=PER_SOURCE_TIMEOUT_SECONDS,
            )
            return hits, None, rejected, True
        raise ValueError("Unsupported source retrieval capability.")
    except (PermissionError, ToolConnectionNotFoundError):
        # Authorization failure aborts; never widen or retry as owner.
        raise PermissionError("Selected source is unavailable.") from None
    except Exception:
        # Provider errors can contain DSNs/SQL parameters. Log them server-side
        # only and return a safe failure marker, never disguise a failed query
        # as empty.
        logger.exception("Source retrieval failed for source %s", target["id"])
        return [], {**failure, "error": "Source retrieval failed or timed out."}, 0, False


async def search_sources(
    user,
    query: str,
    source_hint: str | None = None,
    dataset_ids: list[UUID] | None = None,
    max_sources: int = 6,
    max_catalog_entries: int = 512,
    exclude_source_ids: list[UUID] | None = None,
    include_connections: bool = True,
    top_k: int = 10,
):
    """Search likely sources with the acting user's native permissions.

    Only explicit invocations use this function; automatic session recall never
    starts a live query. Dataset selection excludes connections, whose grants
    are independent. No data is ingested, promoted or written by this operation.
    """
    if user is None:
        raise PermissionError("An acting user is required.")
    if not query.strip() or len(query) > 8000 or not 1 <= top_k <= 100:
        raise ValueError("Invalid source search query or result budget.")

    # Keep ambient session/agent memory out of both routing and SQL generation.
    async def run():
        routing = await route_sources(
            user,
            query,
            source_hint,
            dataset_ids,
            max_sources,
            max_catalog_entries,
            exclude_source_ids,
            include_connections,
        )
        targets = routing.get("targets", [])
        # Every source finishes (or times out) before the result is assembled,
        # so a revoked source still fails the whole call closed, but a slow one
        # costs the others nothing.
        outcomes = await asyncio.gather(
            *(_retrieve_target(user, target, query, top_k) for target in targets),
            return_exceptions=True,
        )
        evidence, errors, searched, rejected, seen = [], [], [], 0, set()
        for target, outcome in zip(targets, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                raise outcome
            hits, error, count, ok = outcome
            rejected += count
            if error is not None:
                errors.append(error)
            if ok:
                searched.append(target)
            for hit in hits:
                # A dataset target and one of its node sets can select the same
                # chunk; keep the first occurrence in routing order.
                key = (hit.get("dataset_id"), hit.get("document_id"), hit.get("text"))
                if hit["retrieval_method"] == "chunks":
                    if key in seen:
                        continue
                    seen.add(key)
                evidence.append(hit)
        return {
            "mode": "search",
            "routing": routing,
            "evidence": evidence,
            "errors": errors,
            "coverage": {
                "complete": False,
                "searched_targets": searched,
                "top_k_per_target": top_k,
                "rejected_unverifiable_hits": rejected,
                "note": "Only selected sources were searched. SQL rows use native connection limits.",
            },
            "next_step": None
            if evidence and not errors
            else (
                "Inspect routing and errors. Browse/read originals for unindexed documents; "
                "database failures are not empty results. Narrow the source or retry where appropriate."
            ),
        }

    # Bound the entire invocation as well as each selected source.
    return await asyncio.wait_for(
        Context().run(lambda: asyncio.create_task(run())), timeout=OVERALL_TIMEOUT_SECONDS
    )
