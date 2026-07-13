"""Map enola snapshot facts to cognee DataPoints and typed graph edges."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from uuid import NAMESPACE_OID, UUID, uuid5

from pydantic import ValidationError

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.enola import (
    normalize_relation,
    parse_enola_snapshot,
    run_enola_generate,
)
from cognee.tasks.code_graph.models import (
    ApiEndpoint,
    CodeModule,
    CodeRepository,
    CodeService,
    CodeSymbol,
    ExternalDependency,
    StorageResource,
)

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_graph")

KIND_TO_MODEL = {
    "module": CodeModule,
    "symbol": CodeSymbol,
    "route": ApiEndpoint,
    "storage": StorageResource,
    "dependency": ExternalDependency,
    "service": CodeService,
}


def fact_node_id(repo: str, kind: str, name: str) -> UUID:
    """Deterministic node id so re-extraction of a repo updates the same nodes.

    Identity is (repo, kind, name): same-named facts of the same kind within a
    repo intentionally collapse into a single node, because enola relations
    reference their targets by bare name. Components are length-prefixed so
    distinct (repo, kind, name) triples can never produce the same hash input.
    """
    key = "enola:" + ":".join(f"{len(part)}:{part}" for part in (repo, kind, name))
    return uuid5(NAMESPACE_OID, key)


def _fact_repo(fact: dict, fallback_repo: str) -> str:
    repo = fact.get("repo")
    return repo if isinstance(repo, str) and repo else fallback_repo


def _resolve_fallback_repo(facts: List[dict], repo_path: Optional[Union[str, Path]]) -> str:
    """Fallback repo for facts without a 'repo' field.

    Must be identical for map_facts_to_data_points and build_code_graph_edges,
    otherwise edge endpoint ids would not match node ids.
    """
    default = Path(repo_path).name if repo_path else "unknown"
    return next(
        (fact["repo"] for fact in facts if isinstance(fact.get("repo"), str) and fact["repo"]),
        default,
    )


def _describe_fact(kind: str, props: dict) -> Optional[str]:
    scalar_props = {
        key: value
        for key, value in (props or {}).items()
        if isinstance(value, (str, int, float, bool))
    }
    if not scalar_props:
        return None
    props_summary = ", ".join(f"{key}={value}" for key, value in sorted(scalar_props.items()))
    return f"{kind}: {props_summary}"


def map_facts_to_data_points(
    facts: List[dict],
    repo_path: Optional[Union[str, Path]] = None,
) -> List[DataPoint]:
    """Map parsed enola facts to DataPoints, prepending one CodeRepository per repo."""
    fallback_repo = _resolve_fallback_repo(facts, repo_path)

    repositories: Dict[str, CodeRepository] = {}

    def _get_repository(repo: str) -> CodeRepository:
        if repo not in repositories:
            repositories[repo] = CodeRepository(
                id=fact_node_id(repo, "repository", repo),
                name=repo,
                path=str(repo_path) if repo_path and repo == fallback_repo else repo,
            )
        return repositories[repo]

    # Always create the primary repository node, even for an empty snapshot.
    _get_repository(fallback_repo)

    entities: List[DataPoint] = []
    skipped_facts = 0

    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        model = KIND_TO_MODEL.get(kind) if isinstance(kind, str) else None

        if model is None or not isinstance(name, str) or not name:
            skipped_facts += 1
            logger.warning("Skipping fact with unknown kind or missing name: %s", fact)
            continue

        repo = _fact_repo(fact, fallback_repo)
        props = fact.get("props")
        if not isinstance(props, dict):
            props = {}
        file_path = fact.get("file")
        line = fact.get("line")

        fields: Dict[str, Any] = {
            "id": fact_node_id(repo, kind, name),
            "name": name,
            "file_path": file_path if isinstance(file_path, str) else None,
            "line": line if isinstance(line, int) and not isinstance(line, bool) else None,
            "repo": repo,
            "description": _describe_fact(kind, props),
            "part_of": _get_repository(repo),
        }
        if model is CodeSymbol:
            fields["symbol_kind"] = props.get("symbol_kind")

        try:
            entities.append(model(**fields))
        except ValidationError:
            skipped_facts += 1
            logger.warning("Skipping fact with invalid field types: %s", fact)

    if skipped_facts:
        logger.warning("Skipped %d fact(s) that could not be mapped to DataPoints.", skipped_facts)

    return list(repositories.values()) + entities


def build_code_graph_edges(
    facts: List[dict],
    repo_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[tuple], int]:
    """Resolve typed relations between facts into explicit graph edge tuples.

    Relation targets are matched by fact name within the same snapshot; on
    ambiguity same-repo targets are preferred, then the relation is skipped.
    repo_path must be the same value given to map_facts_to_data_points so that
    edge endpoint ids match the node ids. Returns (edges, skipped_count).
    """
    fallback_repo = _resolve_fallback_repo(facts, repo_path)

    def _is_mappable(kind: Any, name: Any) -> bool:
        return (
            isinstance(kind, str) and kind in KIND_TO_MODEL and isinstance(name, str) and name != ""
        )

    name_index: Dict[str, set] = {}
    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if not _is_mappable(kind, name):
            continue
        name_index.setdefault(name, set()).add((_fact_repo(fact, fallback_repo), kind))

    edges: List[tuple] = []
    seen_edges = set()
    skipped = 0

    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if not _is_mappable(kind, name):
            continue

        source_repo = _fact_repo(fact, fallback_repo)
        source_id = str(fact_node_id(source_repo, kind, name))

        for relation in fact.get("relations") or []:
            normalized = normalize_relation(relation)
            if normalized is None:
                skipped += 1
                logger.warning("Skipping relation that could not be normalized: %s", relation)
                continue

            relationship_name, target_name = normalized
            candidates = sorted(name_index.get(target_name, ()))
            if len(candidates) > 1:
                candidates = [candidate for candidate in candidates if candidate[0] == source_repo]
            if len(candidates) != 1:
                skipped += 1
                logger.warning(
                    "Skipping relation '%s' from '%s': target '%s' is %s.",
                    relationship_name,
                    name,
                    target_name,
                    "ambiguous" if candidates else "unresolved",
                )
                continue

            target_repo, target_kind = candidates[0]
            target_id = str(fact_node_id(target_repo, target_kind, target_name))

            edge_key = (source_id, target_id, relationship_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edges.append(
                (
                    source_id,
                    target_id,
                    relationship_name,
                    {
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                        "relationship_name": relationship_name,
                        "edge_text": relationship_name.replace("_", " "),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

    return edges, skipped


async def extract_code_graph(
    data: Any = None,
    repo_path: Optional[Union[str, Path]] = None,
    snapshot_dir: Optional[Union[str, Path]] = None,
    timeout: float = 600.0,
) -> List[DataPoint]:
    """Run enola on repo_path (or reuse an existing snapshot) and return DataPoints.

    The returned list composes with the add_data_points task downstream. Typed
    relations are persisted separately by add_code_graph_edges, which re-reads
    the same snapshot after the nodes exist in the graph.
    """
    # When used as the first pipeline task, the pipeline payload arrives as the
    # first positional argument; accept a repo path there, ignore anything else.
    if repo_path is None and isinstance(data, (str, Path)):
        repo_path = data

    if snapshot_dir is None:
        if repo_path is None:
            raise ValueError("extract_code_graph requires repo_path or snapshot_dir.")
        snapshot_dir = await run_enola_generate(repo_path, timeout=timeout)

    facts, receipt = parse_enola_snapshot(snapshot_dir)

    if receipt:
        logger.info(
            "enola snapshot provenance: version=%s snapshot_id=%s",
            receipt.get("enola_version"),
            receipt.get("snapshot_id"),
        )

    data_points = map_facts_to_data_points(facts, repo_path=repo_path)
    logger.info("Mapped %d enola fact(s) to %d data point(s).", len(facts), len(data_points))
    return data_points


async def add_code_graph_edges(
    data_points: List[DataPoint],
    repo_path: Optional[Union[str, Path]] = None,
    snapshot_dir: Optional[Union[str, Path]] = None,
    ctx: Optional["PipelineContext"] = None,
) -> List[DataPoint]:
    """Insert typed relation edges (calls/imports/...) after add_data_points ran.

    Relation names are dynamic, so they cannot be expressed as DataPoint field
    references; instead they are written directly through the graph engine,
    following the extract_dlt_fk_edges precedent. Passthrough: returns
    data_points unchanged.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    if snapshot_dir is None:
        if repo_path is None:
            raise ValueError("add_code_graph_edges requires repo_path or snapshot_dir.")
        snapshot_dir = Path(repo_path) / ".enola"

    facts, _receipt = parse_enola_snapshot(snapshot_dir)
    edges, skipped = build_code_graph_edges(facts, repo_path=repo_path)
    logger.info("Resolved %d code graph edge(s), skipped %d.", len(edges), skipped)

    if edges:
        graph_engine = await get_graph_engine()
        await graph_engine.add_edges(edges)

        # Register the edges in the relational rollback ledger (when a pipeline
        # context with a persisted data item is available) so pipeline rollback
        # can clean them up. Custom pipelines may use arbitrary payloads, such
        # as the repository path used by the code graph example.
        data_item = getattr(ctx, "data_item", None)
        data_id = getattr(data_item, "id", None)
        if (
            ctx is not None
            and getattr(ctx, "user", None) is not None
            and getattr(ctx, "dataset", None) is not None
            and data_id is not None
            and getattr(ctx, "pipeline_run_id", None) is not None
        ):
            from cognee.modules.graph.methods import upsert_edges

            await upsert_edges(
                edges,
                tenant_id=ctx.user.tenant_id,
                user_id=ctx.user.id,
                dataset_id=ctx.dataset.id,
                data_id=data_id,
                pipeline_run_id=ctx.pipeline_run_id,
            )

    return data_points


def get_code_graph_tasks(
    repo_path: Union[str, Path],
    snapshot_dir: Optional[Union[str, Path]] = None,
    timeout: float = 600.0,
) -> List[Task]:
    """Build the ordered task list for the enola code graph pipeline."""
    from cognee.tasks.storage import add_data_points

    return [
        # EXTRACT: run enola and map its facts to DataPoints
        Task(
            extract_code_graph,
            repo_path=repo_path,
            snapshot_dir=snapshot_dir,
            timeout=timeout,
        ),
        # LOAD: persist nodes and embeddings to graph/vector DBs
        Task(add_data_points),
        # LOAD: persist the typed relations as explicit graph edges
        Task(add_code_graph_edges, repo_path=repo_path, snapshot_dir=snapshot_dir),
    ]
