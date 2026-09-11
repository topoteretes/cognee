"""Activity & telemetry endpoints.

Exposes `pipeline_runs` history — pipeline runs and, since SDK-399, one
record per non-pipeline operation — plus in-memory OTEL spans, so the
frontend can render an activity timeline and trace viewer.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from cognee.modules.users.models import User
from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
from cognee.modules.users.methods.get_visible_user_ids import get_visible_user_ids
from cognee.modules.users.permissions.methods.get_permitted_dataset_ids import (
    get_permitted_dataset_ids,
)
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)


def get_activity_router() -> APIRouter:
    router = APIRouter()

    @router.get("/pipeline-runs")
    async def get_pipeline_runs(
        dataset_id: Optional[UUID] = Query(
            None,
            description=(
                "Restrict the feed to a single dataset. When given, a missing read "
                "permission on that dataset is a 403 rather than an empty list."
            ),
        ),
        pipeline_name: Optional[str] = Query(
            None,
            description=(
                "Return only rows whose pipeline_name matches exactly. Operation "
                "records carry no pipeline_name, so this excludes them too — use it to "
                "stop a specific pipeline's history being crowded off the page by "
                "unrelated operation records."
            ),
        ),
        limit: int = Query(50, ge=1, le=500, description="Page size (max 500)."),
        offset: int = Query(0, ge=0, description="Rows to skip for pagination."),
        user: User = Depends(get_authenticated_user),
    ):
        """Recent `pipeline_runs` rows, newest first, with dataset owner info.

        The table records both pipeline runs and — since SDK-399 — one row per
        non-pipeline operation (`search`, `recall`, `remember`, `forget`,
        `delete`, `prune`). Use the **kind** field to tell them apart:

        - `"pipeline"` — a pipeline run (`pipeline_name` is set).
        - `"operation"` — a single-row operation record (`pipeline_name` and
          `status` are NULL, so these are invisible to status-based readers).

        ## Request Parameters
        - **dataset_id** (Optional[UUID]): Restrict to one dataset (403 if not readable).
        - **pipeline_name** (Optional[str]): Exact-match filter; also excludes
          operation records, which have no `pipeline_name`.
        - **limit** (int): Page size, 1-500 (default: 50).
        - **offset** (int): Rows to skip for pagination (default: 0).

        Results are a bare JSON array, not a paged envelope. This endpoint has
        always returned a top-level array, so wrapping it in a `{"runs": [...],
        "total": N}` envelope would break every existing caller — hence no
        `total`. `len(results) == limit` means another page may exist.

        ## Visibility
        Without `dataset_id`: rows owned by the caller (and their child agents),
        plus rows on any dataset shared with them. Operation records for
        `recall`, `prune`, and multi-dataset `search` carry no `dataset_id`, so a
        dataset-only filter would omit them entirely.

        ## Response
        A JSON array. Alongside the original `id`, `pipeline_name`, `status`,
        `dataset_id`, `dataset_name`, `owner_id`, `owner_email`, `created_at`
        and `pipeline_run_id` keys, each row carries the SDK-399 operation
        columns. **Every one of them is nullable**: rows written before SDK-399
        were not backfilled, and each writer sets only the subset it knows.

        - **kind** (str): `"pipeline"` or `"operation"` (never null).
        - **operation_name** (str|null): Operation name; for pipeline rows this
          mirrors `pipeline_name`, so it does *not* distinguish the two kinds.
        - **origin** (str|null): Initiating surface — `sdk`/`api`/`cli`/`mcp`/`background`.
        - **outcome** (str|null): `"succeeded"` / `"failed"`. NULL on non-terminal rows.
          **Read together with `background`**: when `background` is true, a
          `"succeeded"` outcome means the work was *accepted and started*, not that
          it finished. Treating those rows as completions inflates any success-rate
          or cost figure computed from this feed.
        - **background** (bool|null): True when the call launched background work.
          NULL means not applicable / not recorded.
        - **error_class** (str|null): Exception class name when `outcome="failed"`.
        - **tokens_in** / **tokens_out** (int|null): Provider-billed token counts.
          NULL means *not measured*; `0` means *measured zero* — do not conflate.
        - **started_at** / **ended_at** (str|null): ISO-8601 timestamps.
        - **user_id** (str|null): Triggering user.
        - **session_id** (str|null): Session-cache id; joins `session_model_usage`.
        - **parent_operation_id** (str|null): Parent's `pipeline_run_id`.

        ## Aggregation caveats (append-only table)
        Rows are append-only, so totals must not be summed naively:

        1. A pipeline run emits several rows sharing one `pipeline_run_id`
           (initiated → started → terminal). Only the terminal row carries
           `outcome` and `tokens_*`. Deduplicate by `pipeline_run_id` before
           summing, or you will multiply-count.
        2. `parent_operation_id` forms a tree whose token counts already chain
           into the parent. Summing across levels double-counts; sum one level.
        """
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.pipelines.models import PipelineRun
        from cognee.modules.data.models.Dataset import Dataset
        from cognee.modules.users.models import User
        from sqlalchemy import select, outerjoin, or_

        if dataset_id is not None:
            # Explicit dataset request stays strict, and needs no empty-result
            # branch: for a dataset the caller cannot read this helper raises
            # rather than returning [], and that PermissionDeniedError
            # propagates as the original 403 instead of a silently empty feed.
            permitted_datasets = await get_specific_user_permission_datasets(
                user.id, "read", [dataset_id]
            )
            permitted_dataset_id_set = {ds.id for ds in permitted_datasets}
            visibility = PipelineRun.dataset_id.in_(permitted_dataset_id_set)
        else:
            # Same "mine, or shared with me" predicate the sessions router
            # applies to session_records. get_permitted_dataset_ids already
            # translates "no readable datasets" into an empty list, so there is
            # no PermissionDeniedError to catch here.
            visible_user_ids = await get_visible_user_ids(user.id)
            permitted_dataset_id_set = set(await get_permitted_dataset_ids(user.id))
            visibility_terms = [PipelineRun.user_id.in_(visible_user_ids)]
            if permitted_dataset_id_set:
                visibility_terms.append(PipelineRun.dataset_id.in_(permitted_dataset_id_set))
            visibility = or_(*visibility_terms)

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            # Join pipeline runs → dataset → owner user for agent attribution
            stmt = (
                select(
                    PipelineRun,
                    Dataset.name.label("ds_name"),
                    Dataset.owner_id,
                    User.email.label("owner_email"),
                )
                .select_from(
                    outerjoin(PipelineRun, Dataset, PipelineRun.dataset_id == Dataset.id).outerjoin(
                        User, Dataset.owner_id == User.id
                    )
                )
                .where(visibility)
                # id breaks created_at ties. Without it, OFFSET paging over a
                # table that gains a row on every operation re-serves rows the
                # client already saw, because equal timestamps order arbitrarily.
                .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
                .limit(limit)
                .offset(offset)
            )
            if pipeline_name is not None:
                stmt = stmt.where(PipelineRun.pipeline_name == pipeline_name)

            result = await session.execute(stmt)
            rows = result.all()

        return [
            {
                "id": str(run.id),
                # Operation rows leave pipeline_name NULL; every pipeline writer
                # sets it. Derived here so clients need not know that convention.
                "kind": "pipeline" if run.pipeline_name is not None else "operation",
                "pipeline_name": run.pipeline_name,
                "status": run.status.value if run.status else None,
                "dataset_id": str(run.dataset_id) if run.dataset_id else None,
                # The row itself is visible via the user_id term even when its
                # dataset_id is not in permitted_dataset_id_set (the caller has
                # write, not read, on it, or read was revoked since the run).
                # The joined dataset name and owner email are read-gated
                # information, unlike the id the caller already knows from
                # having triggered the run — withhold them for that case.
                "dataset_name": ds_name if run.dataset_id in permitted_dataset_id_set else None,
                "owner_id": str(owner_id)
                if owner_id and run.dataset_id in permitted_dataset_id_set
                else None,
                "owner_email": owner_email if run.dataset_id in permitted_dataset_id_set else None,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "pipeline_run_id": str(run.pipeline_run_id) if run.pipeline_run_id else None,
                # Operation-record columns (SDK-399), all nullable — see docstring.
                "operation_name": run.operation_name,
                "origin": run.origin,
                "outcome": run.outcome,
                # Qualifies outcome: with background true, "succeeded" means
                # accepted-and-started, not finished. See docstring.
                "background": run.background,
                "error_class": run.error_class,
                # Passed through untouched: 0 is "measured zero" and NULL is
                # "not measured", so a truthiness check would destroy the
                # distinction the schema deliberately encodes.
                "tokens_in": run.tokens_in,
                "tokens_out": run.tokens_out,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "ended_at": run.ended_at.isoformat() if run.ended_at else None,
                "user_id": str(run.user_id) if run.user_id else None,
                "session_id": run.session_id,
                "parent_operation_id": str(run.parent_operation_id)
                if run.parent_operation_id
                else None,
            }
            for run, ds_name, owner_id, owner_email in rows
        ]

    @router.get("/spans")
    async def get_spans(user: User = Depends(get_authenticated_user)):
        """Return in-memory OTEL spans from the CogneeSpanExporter buffer."""
        try:
            from cognee.modules.observability.tracing import get_exporter
            from cognee.modules.observability.trace_context import is_tracing_enabled

            # Lazily initialize tracing if enabled but not yet set up
            # (exporter is None until first span or explicit enable_tracing call)
            is_tracing_enabled()

            exporter = get_exporter()
            if exporter is None:
                return []

            all_traces = exporter.get_all_traces()

            result = []
            for trace_id, spans in all_traces.items():
                root = spans[0] if spans else None
                duration = max((s.get("duration_ms", 0) for s in spans), default=0)
                result.append(
                    {
                        "trace_id": trace_id,
                        "root_name": root.get("name") if root else None,
                        "duration_ms": duration,
                        "span_count": len(spans),
                        "status": root.get("status") if root else None,
                        "spans": spans,
                    }
                )

            return result
        except Exception:
            logger.exception("Failed to retrieve activity traces")
            return JSONResponse(
                status_code=500,
                content={"error": "Unable to retrieve activity traces."},
            )

    @router.get("/users")
    async def get_tenant_users(user: User = Depends(get_authenticated_user)):
        """Return users in the current tenant (includes agents as API key users)."""
        try:
            from cognee.modules.users.tenants.methods import get_users_in_tenant

            users = await get_users_in_tenant(user.tenant_id)
            return [
                {
                    "id": str(u.id),
                    "email": u.email,
                    "is_superuser": u.is_superuser,
                    "created_at": u.created_at.isoformat()
                    if hasattr(u, "created_at") and u.created_at
                    else None,
                }
                for u in users
            ]
        except Exception:
            return []

    @router.get("/agents")
    async def get_agents(user: User = Depends(get_authenticated_user)):
        """Return registered agents (users with @cognee.agent emails)."""
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.users.models import User
        from cognee.modules.users.models.UserApiKey import UserApiKey
        from cognee.modules.data.models.Data import Data
        from cognee.modules.search.models.Query import Query
        from sqlalchemy import select, func
        from datetime import datetime, timedelta, timezone

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            # Get all users (agents have @cognee.agent, but show all non-default)
            users_q = select(User).filter(User.is_active.is_(True))  # noqa: E712
            users_result = await session.execute(users_q)
            all_users = users_result.scalars().all()

            # Count API keys per user
            keys_q = select(UserApiKey.user_id, func.count().label("key_count")).group_by(
                UserApiKey.user_id
            )
            keys_result = await session.execute(keys_q)
            key_counts = {str(row.user_id): row.key_count for row in keys_result}

            # Get latest data ingestion per user
            data_created_q = select(Data.owner_id, func.max(Data.created_at).label("ts")).group_by(
                Data.owner_id
            )
            data_created_result = await session.execute(data_created_q)
            last_active_map = {
                str(row.owner_id): row.ts for row in data_created_result if row.owner_id
            }

            # Get latest data access per user
            data_accessed_q = (
                select(Data.owner_id, func.max(Data.last_accessed).label("ts"))
                .filter(Data.last_accessed.isnot(None))
                .group_by(Data.owner_id)
            )
            data_accessed_result = await session.execute(data_accessed_q)
            for row in data_accessed_result:
                if not row.owner_id:
                    continue
                uid = str(row.owner_id)
                if uid not in last_active_map or (row.ts and row.ts > last_active_map[uid]):
                    last_active_map[uid] = row.ts

            # Get latest search query per user
            search_q = select(Query.user_id, func.max(Query.created_at).label("ts")).group_by(
                Query.user_id
            )
            search_result = await session.execute(search_q)
            for row in search_result:
                if not row.user_id:
                    continue
                uid = str(row.user_id)
                if uid not in last_active_map or (row.ts and row.ts > last_active_map[uid]):
                    last_active_map[uid] = row.ts

        now = datetime.now(timezone.utc)
        live_cutoff = now - timedelta(minutes=30)

        agents = []
        for u in all_users:
            email = u.email or ""
            is_agent = email.endswith("@cognee.agent")
            is_default = email == "default_user@example.com"

            # Parse agent type from email
            # Internal email format: "sanitized-name+{parent_user_id}@cognee.agent"
            # The +{parent_user_id} suffix ensures uniqueness across users but
            # must be stripped for display purposes.
            # Legacy agents used "-" throughout, so we fall back to rsplit("-", 1)
            # when no "+" is present and the suffix looks like a hex UUID fragment.
            if is_agent:
                local_part = email.split("@")[0]
                if "+" in local_part:
                    # Current format: name+user_id
                    display_name, agent_short_id = local_part.rsplit("+", 1)
                elif "-" in local_part:
                    # Legacy format: name-part-of-uuid — strip UUID suffix
                    prefix, suffix = local_part.rsplit("-", 1)
                    if len(suffix) >= 8 and all(c in "0123456789abcdef" for c in suffix):
                        display_name, agent_short_id = prefix, suffix
                    else:
                        display_name, agent_short_id = local_part, ""
                else:
                    display_name, agent_short_id = local_part, ""
                agent_type = display_name.replace("-", " ").replace("_", " ")
            else:
                agent_type = "Human User" if is_default else email.split("@")[0]
                agent_short_id = ""

            api_key_count = key_counts.get(str(u.id), 0)
            last_active = last_active_map.get(str(u.id))

            # Determine status based on actual activity
            if last_active and last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=timezone.utc)

            if last_active and last_active > live_cutoff:
                status = "LIVE"
            elif last_active:
                status = "INACTIVE"
            else:
                status = "NEVER_CONNECTED"

            agents.append(
                {
                    "id": str(u.id),
                    "email": email,
                    "agent_type": agent_type,
                    "agent_short_id": agent_short_id,
                    "is_agent": is_agent,
                    "is_default": is_default,
                    "status": status,
                    "api_key_count": api_key_count,
                    "created_at": u.created_at.isoformat()
                    if hasattr(u, "created_at") and u.created_at
                    else None,
                    "last_active": last_active.isoformat() if last_active else None,
                }
            )

        return agents

    @router.get("/export/{dataset_id}")
    async def export_dataset_markdown(dataset_id: UUID, user=Depends(get_authenticated_user)):
        """Export a dataset's knowledge graph as a Markdown memory report.

        ## Path Parameters
        - **dataset_id** (UUID): UUID of the dataset (from GET /api/v1/datasets).
        """
        from fastapi.responses import Response
        from cognee.modules.data.models.Dataset import Dataset
        from cognee.modules.data.models.Data import Data
        from cognee.modules.graph.methods import get_formatted_graph_data
        from cognee.infrastructure.databases.relational import get_relational_engine
        from sqlalchemy import select
        from datetime import datetime, timezone

        dataset_ids = await get_specific_user_permission_datasets(user.id, "read", [dataset_id])
        dataset_id = dataset_ids[0].id

        db_engine = get_relational_engine()

        # Get dataset info
        async with db_engine.get_async_session() as session:
            ds_result = await session.execute(select(Dataset).filter(Dataset.id == dataset_id))
            dataset = ds_result.scalar_one_or_none()
            if not dataset:
                return Response(content="Dataset not found", status_code=404)

            # Get documents (dataset-scoped rows)
            docs_result = await session.execute(select(Data).filter(Data.dataset_id == dataset_id))
            docs = docs_result.scalars().all()

        # Get graph data
        try:
            graph = await get_formatted_graph_data(dataset_id, user)
            nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
            edges = graph.get("edges", []) if isinstance(graph, dict) else []
        except Exception:
            nodes, edges = [], []

        # Build markdown
        now = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
        ds_name = dataset.name or str(dataset_id)

        # Categorize nodes
        entities = [n for n in nodes if n.get("type") == "Entity"]
        summaries = [n for n in nodes if n.get("type") == "TextSummary"]
        other_nodes = [
            n
            for n in nodes
            if n.get("type") not in ("Entity", "TextSummary", "DocumentChunk", "TextDocument")
        ]

        lines = []
        lines.append(f"# Dataset: {ds_name}")
        lines.append("")
        lines.append(
            f"Exported: {now} | {len(docs)} documents | {len(entities)} entities | {len(edges)} relationships"
        )
        lines.append("")

        # Summaries
        if summaries:
            lines.append("## Summaries")
            lines.append("")
            for s in summaries:
                text = s.get("properties", {}).get("text", "")
                if text:
                    lines.append(f"> {text}")
                    lines.append("")

        # Entities
        if entities:
            lines.append("## Entities")
            lines.append("")
            lines.append("| Entity | Description |")
            lines.append("|--------|-------------|")
            for e in entities:
                label = e.get("label", "?")
                desc = e.get("properties", {}).get("description", "")
                # Escape pipes in markdown
                label = label.replace("|", "\\|")
                desc = desc.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {label} | {desc} |")
            lines.append("")

        # Relationships
        if edges:
            # Build label lookup
            node_labels = {n.get("id"): n.get("label", n.get("id", "?")[:12]) for n in nodes}
            lines.append("## Relationships")
            lines.append("")
            lines.append("| Source | Relationship | Target |")
            lines.append("|--------|-------------|--------|")
            for e in edges:
                src = node_labels.get(e.get("source"), e.get("source", "?")[:12])
                tgt = node_labels.get(e.get("target"), e.get("target", "?")[:12])
                rel = e.get("label", "related_to")
                src = src.replace("|", "\\|")
                tgt = tgt.replace("|", "\\|")
                rel = rel.replace("|", "\\|")
                lines.append(f"| {src} | {rel} | {tgt} |")
            lines.append("")

        # Documents
        if docs:
            lines.append("## Documents")
            lines.append("")
            for d in docs:
                name = d.name or "unnamed"
                ext = (d.extension or "").upper()
                created = d.created_at.strftime("%b %d, %Y") if d.created_at else "?"
                lines.append(f"- **{name}** ({ext}, {created})")
            lines.append("")

        # Other node types
        if other_nodes:
            lines.append("## Other Nodes")
            lines.append("")
            for n in other_nodes:
                ntype = n.get("type", "?")
                label = n.get("label", "?")
                lines.append(f"- [{ntype}] {label}")
            lines.append("")

        markdown = "\n".join(lines)
        filename = f"{ds_name}-memory-export.md"

        return Response(
            content=markdown,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
