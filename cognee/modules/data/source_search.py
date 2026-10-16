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


async def _document_evidence(user, target, query, top_k, seen):
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
                raise PermissionError("Evidence is outside the selected source.")
            key = (target["dataset_id"], document, payload.get("text"))
            if key in seen:
                continue
            seen.add(key)
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
        evidence, errors, searched, seen, rejected = [], [], [], set(), 0
        for target in routing.get("targets", []):
            method = target.get("retrieval_method", "chunks")
            try:

                async def retrieve(target=target, method=method):
                    nonlocal rejected
                    if method == "sql":
                        # Re-resolves the same caller's connection at execution.
                        # Native guards enforce feature gate, read-only SQL,
                        # table allowlists, statement timeout and row limits.
                        result = await run_text_to_sql(user.id, target["connection"], query)
                        if not result.success:
                            errors.append(
                                {
                                    "source_id": target["id"],
                                    "retrieval_method": method,
                                    "error": "Native database retrieval failed.",
                                }
                            )
                            return False
                        evidence.append(
                            {
                                "retrieval_method": "sql",
                                "source_target": target["name"],
                                "queried_at": datetime.now(timezone.utc).isoformat(),
                                "success": True,
                                "text": result.render_text(),
                                "structured": result.structured(),
                            }
                        )
                    elif method == "chunks":
                        hits, count = await _document_evidence(user, target, query, top_k, seen)
                        evidence.extend(hits)
                        rejected += count
                    else:
                        raise ValueError("Unsupported source retrieval capability.")

                if await asyncio.wait_for(retrieve(), timeout=90) is not False:
                    searched.append(target)
            except (PermissionError, ToolConnectionNotFoundError):
                # Authorization failure aborts; never widen or retry as owner.
                raise PermissionError("Selected source is unavailable.") from None
            except Exception:  # noqa: BLE001 — sanitize errors at the retrieval boundary
                # Provider errors can contain DSNs/SQL parameters. Return a
                # safe failure marker, never disguise a failed query as empty.
                errors.append(
                    {
                        "source_id": target["id"],
                        "retrieval_method": method,
                        "error": "Source retrieval failed or timed out.",
                    }
                )
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
    return await asyncio.wait_for(Context().run(lambda: asyncio.create_task(run())), timeout=270)
