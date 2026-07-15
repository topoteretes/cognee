"""Map enola snapshot facts to cognee DataPoints and typed graph edges."""

from datetime import datetime, timezone
from pathlib import Path
import posixpath
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
    CodeTestReference,
    CodeFileReference,
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
    "test_ref": CodeTestReference,
    "file_ref": CodeFileReference,
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
            "kind": kind,
            "file_path": file_path if isinstance(file_path, str) else None,
            "line": line if isinstance(line, int) and not isinstance(line, bool) else None,
            "repo": repo,
            "description": _describe_fact(kind, props),
            "fact_properties": props,
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

    valid_facts = []
    name_index: Dict[str, set] = {}
    fact_index: Dict[Tuple[str, str, str], dict] = {}
    module_index: Dict[Tuple[str, str], dict] = {}
    module_path_repos: Dict[str, set[str]] = {}
    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if not _is_mappable(kind, name):
            continue
        repo = _fact_repo(fact, fallback_repo)
        valid_facts.append((fact, repo))
        name_index.setdefault(name, set()).add((repo, kind))
        fact_index.setdefault((repo, kind, name), fact)
        if kind == "module":
            module_index.setdefault((repo, name), fact)
            props = fact.get("props")
            module_path = props.get("modulePath") if isinstance(props, dict) else None
            if isinstance(module_path, str) and module_path:
                module_path_repos.setdefault(module_path, set()).add(repo)

    edges: List[tuple] = []
    seen_edges = set()
    skipped = 0

    def _resolve_target(
        target_name: str,
        source_repo: str,
        allowed_repos: Optional[set[str]] = None,
    ) -> Optional[Tuple[str, str, str]]:
        candidates = set(name_index.get(target_name, ()))
        if allowed_repos is not None:
            candidates = {candidate for candidate in candidates if candidate[0] in allowed_repos}
        same_repo = {candidate for candidate in candidates if candidate[0] == source_repo}
        if len(same_repo) == 1:
            repo, kind = next(iter(same_repo))
            return repo, kind, target_name
        if len(candidates) == 1:
            repo, kind = next(iter(candidates))
            return repo, kind, target_name
        return None

    def _add_edge(
        source: Tuple[str, str, str],
        target: Tuple[str, str, str],
        relationship_name: str,
    ) -> None:
        source_repo, source_kind, source_name = source
        target_repo, target_kind, target_name = target
        source_id = fact_node_id(source_repo, source_kind, source_name)
        target_id = fact_node_id(target_repo, target_kind, target_name)
        edge_key = (source_id, target_id, relationship_name)
        if edge_key in seen_edges:
            return
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

    # Persist every explicit relation. For unresolved Go calls, mirror Enola's
    # module-path normalization before giving up on the target.
    for fact, source_repo in valid_facts:
        kind = fact["kind"]
        name = fact["name"]

        for relation in fact.get("relations") or []:
            normalized = normalize_relation(relation)
            if normalized is None:
                skipped += 1
                logger.warning("Skipping relation that could not be normalized: %s", relation)
                continue

            relationship_name, target_name = normalized
            target = _resolve_target(target_name, source_repo)
            if target is None and relationship_name == "calls" and not name_index.get(target_name):
                for module_path in sorted(
                    module_path_repos, key=lambda value: (-len(value), value)
                ):
                    if not target_name.startswith(module_path):
                        continue
                    provider_repos = module_path_repos[module_path]
                    if len(provider_repos) != 1:
                        continue
                    suffix = target_name[len(module_path) :]
                    if suffix.startswith("/"):
                        normalized_target = suffix[1:]
                    elif suffix.startswith("."):
                        normalized_target = "." + suffix
                    else:
                        continue
                    if normalized_target:
                        target = _resolve_target(
                            normalized_target,
                            source_repo,
                            allowed_repos=provider_repos,
                        )
                    if target is not None:
                        break

            if target is None:
                skipped += 1
                logger.warning(
                    "Skipping relation '%s' from '%s': target '%s' is %s.",
                    relationship_name,
                    name,
                    target_name,
                    "ambiguous" if name_index.get(target_name) else "unresolved",
                )
                continue

            target_repo, target_kind, resolved_target_name = target
            _add_edge(
                (source_repo, kind, name),
                (target_repo, target_kind, resolved_target_name),
                relationship_name,
            )

    # Enola's query graph connects a dependency import to the modules which
    # contain each side. Materialize the same bridge so Cognee traversals can
    # move between modules without re-reading the snapshot.
    modules_by_name: Dict[str, list[Tuple[str, dict]]] = {}
    for (repo, module_name), module_fact in module_index.items():
        modules_by_name.setdefault(module_name, []).append((repo, module_fact))

    for fact, source_repo in valid_facts:
        if fact["kind"] != "dependency" or not isinstance(fact.get("file"), str):
            continue
        file_path = fact["file"]
        source_directories = [posixpath.dirname(file_path) or "."]
        repo_prefix = source_repo.rstrip("/") + "/"
        if file_path.startswith(repo_prefix):
            source_directories.append(posixpath.dirname(file_path[len(repo_prefix) :]) or ".")
        source_module = next(
            (
                module_index[(source_repo, directory)]
                for directory in source_directories
                if (source_repo, directory) in module_index
            ),
            None,
        )
        if source_module is None:
            continue

        for relation in fact.get("relations") or []:
            normalized = normalize_relation(relation)
            if normalized is None or normalized[0] != "imports":
                continue
            target_name = normalized[1]
            target_module: Optional[Tuple[str, dict]] = None
            candidate_name = target_name
            while candidate_name:
                local = module_index.get((source_repo, candidate_name))
                if local is not None:
                    target_module = (source_repo, local)
                    break
                global_matches = modules_by_name.get(candidate_name, [])
                if len(global_matches) == 1:
                    target_module = global_matches[0]
                    break
                parent = posixpath.dirname(candidate_name)
                if parent in (candidate_name, "."):
                    break
                candidate_name = parent

            if target_module is None:
                continue
            target_repo, target_fact = target_module
            if source_repo == target_repo and source_module["name"] == target_fact["name"]:
                continue
            _add_edge(
                (source_repo, "module", source_module["name"]),
                (target_repo, "module", target_fact["name"]),
                "imports",
            )

    # Enola derives type membership from qualified symbol names. Persisting the
    # edge makes type-level impact analysis include method/function callers.
    owner_symbol_kinds = {"struct", "interface", "class", "type"}
    for fact, source_repo in valid_facts:
        if fact["kind"] != "symbol" or "." not in fact["name"]:
            continue
        props = fact.get("props")
        symbol_kind = props.get("symbol_kind") if isinstance(props, dict) else None
        if symbol_kind not in {"method", "function"}:
            continue
        owner_name = fact["name"].rsplit(".", 1)[0]
        owner = fact_index.get((source_repo, "symbol", owner_name))
        owner_props = owner.get("props") if owner is not None else None
        if (
            not isinstance(owner_props, dict)
            or owner_props.get("symbol_kind") not in owner_symbol_kinds
        ):
            continue
        _add_edge(
            (source_repo, "symbol", owner_name),
            (source_repo, "symbol", fact["name"]),
            "has_method",
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


def _pipeline_data_id(ctx: Optional["PipelineContext"] = None) -> Any:
    """Return a stable id for a persisted pipeline item, if it has one."""
    data_item = getattr(ctx, "data_item", None)
    data_id = getattr(data_item, "id", None)
    return data_id if data_id is not None else getattr(data_item, "data_id", None)


def _invalidate_code_graph_snapshot(ctx: Optional["PipelineContext"] = None) -> None:
    """Invalidate the exact dataset cache even if a caller omitted DB context."""
    from cognee.modules.retrieval.code_retriever import invalidate_code_graph_snapshot_cache

    dataset = getattr(ctx, "dataset", None)
    dataset_id = getattr(dataset, "id", None)
    if dataset_id is None:
        invalidate_code_graph_snapshot_cache()
    else:
        invalidate_code_graph_snapshot_cache(dataset_id=dataset_id)


async def add_code_graph_data_points(
    data_points: List[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    graph_only: bool = True,
) -> List[DataPoint]:
    """Store code graph nodes while allowing a repository path payload.

    A custom-pipeline payload may be any value, but the storage rollback ledger
    requires a persisted data item id. Preserve the full context when one is
    available and otherwise store without ledger provenance. graph_only keeps
    the deterministic default free of embedding calls; set it to False to also
    build vector indexes for completion-based search types.
    """
    from cognee.tasks.storage.add_data_points import add_data_points

    storage_ctx = ctx if _pipeline_data_id(ctx) is not None else None
    try:
        result = await add_data_points(data_points, ctx=storage_ctx, graph_only=graph_only)
    finally:
        # Storage can fail after a partial graph write. Invalidate even on an
        # exception so no pre-write snapshot survives that ambiguous outcome.
        _invalidate_code_graph_snapshot(ctx)
    return result


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

    try:
        if edges:
            graph_engine = await get_graph_engine()
            await graph_engine.add_edges(edges)

            # Register the edges in the relational rollback ledger (when a pipeline
            # context with a persisted data item is available) so pipeline rollback
            # can clean them up. Custom pipelines may use arbitrary payloads, such
            # as the repository path used by the code graph example.
            data_id = _pipeline_data_id(ctx)
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
    finally:
        # Direct edge writes and rollback-ledger writes may partially succeed.
        _invalidate_code_graph_snapshot(ctx)
    return data_points


def get_code_graph_tasks(
    repo_path: Union[str, Path],
    snapshot_dir: Optional[Union[str, Path]] = None,
    timeout: float = 600.0,
    index_vectors: bool = False,
) -> List[Task]:
    """Build the ordered task list for the enola code graph pipeline.

    index_vectors is opt-in because SearchType.CODE uses graph indexes only.
    Enable it when the same code facts must also feed semantic/completion
    retrievers, which may require an embedding provider API key.
    """
    return [
        # EXTRACT: run enola and map its facts to DataPoints
        Task(
            extract_code_graph,
            repo_path=repo_path,
            snapshot_dir=snapshot_dir,
            timeout=timeout,
        ),
        # LOAD: persist graph nodes; vector indexing is explicitly opt-in
        Task(add_code_graph_data_points, graph_only=not index_vectors),
        # LOAD: persist the typed relations as explicit graph edges
        Task(add_code_graph_edges, repo_path=repo_path, snapshot_dir=snapshot_dir),
    ]
