"""Activity & telemetry endpoints.

Exposes pipeline run history and in-memory OTEL spans so the frontend
can render an activity timeline and trace viewer.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Query, Depends
from cognee.modules.users.models import User
from cognee.modules.users.methods.get_authenticated_user import get_authenticated_user
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.permissions.methods.get_specific_user_permission_datasets import (
    get_specific_user_permission_datasets,
)


def get_activity_router() -> APIRouter:
    router = APIRouter()

    @router.get("/pipeline-runs")
    async def get_pipeline_runs(
        dataset_id: Optional[UUID] = Query(None), user: User = Depends(get_authenticated_user)
    ):
        """Return recent pipeline runs with dataset owner info for agent attribution."""
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.pipelines.models import PipelineRun
        from cognee.modules.data.models.Dataset import Dataset
        from cognee.modules.users.models import User
        from sqlalchemy import select, outerjoin

        try:
            permitted_datasets = await get_specific_user_permission_datasets(
                user.id, "read", [dataset_id] if dataset_id else None
            )
        except PermissionDeniedError:
            # For list requests, treat "no accessible datasets" as an empty activity feed.
            # Keep explicit dataset requests strict by propagating the original 403.
            if dataset_id is None:
                return []
            raise

        dataset_ids = [ds.id for ds in permitted_datasets]
        if not dataset_ids:
            return []

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
                .where(PipelineRun.dataset_id.in_(dataset_ids))
                .order_by(PipelineRun.created_at.desc())
                .limit(50)
            )

            result = await session.execute(stmt)
            rows = result.all()

        return [
            {
                "id": str(run.id),
                "pipeline_name": run.pipeline_name,
                "status": run.status.value if run.status else None,
                "dataset_id": str(run.dataset_id) if run.dataset_id else None,
                "dataset_name": ds_name,
                "owner_id": str(owner_id) if owner_id else None,
                "owner_email": owner_email,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "pipeline_run_id": str(run.pipeline_run_id) if run.pipeline_run_id else None,
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
        except Exception as e:
            return {"error": str(e)}

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
        from cognee.modules.pipelines.models import PipelineRun
        from sqlalchemy import select, func, case
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

            # Get latest pipeline run per user (approximate: via dataset ownership)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_q = (
                select(PipelineRun.dataset_id, func.count().label("run_count"))
                .filter(PipelineRun.created_at > cutoff)
                .group_by(PipelineRun.dataset_id)
            )
            await session.execute(recent_q)

        agents = []
        for u in all_users:
            email = u.email or ""
            is_agent = email.endswith("@cognee.agent")
            is_default = email == "default_user@example.com"

            # Parse agent type from email
            if is_agent:
                local_part = email.split("@")[0]
                parts = local_part.rsplit("-", 1)
                agent_type = parts[0].replace("-", " ").replace("_", " ") if parts else local_part
                agent_short_id = parts[1] if len(parts) > 1 else ""
            else:
                agent_type = "Human User" if is_default else email.split("@")[0]
                agent_short_id = ""

            api_key_count = key_counts.get(str(u.id), 0)
            has_recent = api_key_count > 0  # Simplified: has key = potentially active

            agents.append(
                {
                    "id": str(u.id),
                    "email": email,
                    "agent_type": agent_type,
                    "agent_short_id": agent_short_id,
                    "is_agent": is_agent,
                    "is_default": is_default,
                    "status": "LIVE" if has_recent else "INACTIVE",
                    "api_key_count": api_key_count,
                    "created_at": u.created_at.isoformat()
                    if hasattr(u, "created_at") and u.created_at
                    else None,
                }
            )

        return agents

    @router.get("/export/{dataset_id}")
    async def export_dataset_markdown(dataset_id: UUID, user=Depends(get_authenticated_user)):
        """Export a dataset's knowledge graph as a Markdown memory report."""
        from fastapi.responses import Response
        from cognee.modules.data.models.Dataset import Dataset
        from cognee.modules.data.models.DatasetData import DatasetData
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

            # Get documents (join DatasetData → Data)
            docs_result = await session.execute(
                select(Data)
                .join(DatasetData, Data.id == DatasetData.data_id)
                .filter(DatasetData.dataset_id == dataset_id)
            )
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
