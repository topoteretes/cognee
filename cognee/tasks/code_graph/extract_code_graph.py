"""Map enola snapshot facts to cognee DataPoints and typed graph edges."""

import hashlib
import json
import posixpath
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import NAMESPACE_OID, UUID, uuid5

from pydantic import ValidationError

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models.node_set import node_sets_from_names
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.enola import (
    is_enola_id,
    normalize_relation,
    parse_enola_snapshot,
    relation_target_id,
    run_enola_generate,
    snapshot_identity,
)
from cognee.tasks.code_graph.models import (
    ApiEndpoint,
    CodeAssociation,
    CodeExtractionAccount,
    CodeFileReference,
    CodeInsight,
    CodeIntent,
    CodeLintFinding,
    CodeModule,
    CodeRepository,
    CodeService,
    CodeSymbol,
    CodeTestReference,
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
    "insight": CodeInsight,
    "intent": CodeIntent,
    "extraction": CodeExtractionAccount,
    "association": CodeAssociation,
    "lint": CodeLintFinding,
}

# receipt.json fields worth keeping on the repository node. repo_path,
# duration and generated_at-style run specifics that describe the machine
# rather than the snapshot are left out, except generated_at itself, which
# dates the stamped snapshot.
_RECEIPT_PROJECTION_KEYS = (
    "format_version",
    "enola_version",
    "extractor_version",
    "generated_at",
    "git",
    "extractors",
    "fact_count",
    "insight_count",
    "quality",
)

# Ids per delete statement when sweeping stale nodes/edges, mirroring the
# write-side _WRITE_CHUNK_SIZE convention so no single statement can run past
# the subprocess engine's per-call deadline.
_SWEEP_CHUNK_SIZE = 2000


def _is_mappable_fact(kind: Any, name: Any) -> bool:
    """Whether a fact will become a graph node (known kind, non-empty name)."""
    return isinstance(kind, str) and kind in KIND_TO_MODEL and isinstance(name, str) and name != ""


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


def _resolve_fallback_repo(facts: list[dict], repo_path: str | Path | None) -> str:
    """Fallback repo for facts without a 'repo' field.

    Must be identical for map_facts_to_data_points and build_code_graph_edges,
    otherwise edge endpoint ids would not match node ids.
    """
    default = Path(repo_path).name if repo_path else "unknown"
    return next(
        (fact["repo"] for fact in facts if isinstance(fact.get("repo"), str) and fact["repo"]),
        default,
    )


def _describe_fact(kind: str, props: dict) -> str | None:
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
    facts: list[dict],
    repo_path: str | Path | None = None,
    node_set: list[str] | None = None,
) -> list[DataPoint]:
    """Map parsed enola facts to DataPoints, prepending one CodeRepository per repo.

    ``node_set`` (the caller's add()/remember() tag, read off the repo
    manifest's external_metadata by extract_code_repo_graph) tags every
    returned node -- CodeRepository included -- with the same
    belongs_to_set/source_node_set a document gets from classify_documents.
    An untagged call (node_set=None, the per-file CODE route and the legacy
    remember(content_type="code") path) produces byte-identical fact_hash
    values to before this parameter existed.

    The returned CodeRepository's ``last_node_set`` is left at its model
    default (``None``) here -- ``add_code_graph_data_points`` overrides it
    from its own pre-write graph read right before storing, which is the
    only read guaranteed to run immediately before the write. See that
    function's docstring for why the marker is never set at construction
    time.
    """
    fallback_repo = _resolve_fallback_repo(facts, repo_path)

    node_sets = node_sets_from_names(node_set) if node_set else None
    source_node_set = ", ".join(node_set) if node_set else None
    # Sorted so re-tagging with the same names in a different order does not
    # spuriously flip every fact's hash.
    node_set_hash_value = sorted(node_set) if node_set else None

    repositories: dict[str, CodeRepository] = {}

    def _get_repository(repo: str) -> CodeRepository:
        if repo not in repositories:
            repositories[repo] = CodeRepository(
                id=fact_node_id(repo, "repository", repo),
                name=repo,
                path=str(repo_path) if repo_path and repo == fallback_repo else repo,
                belongs_to_set=node_sets,
                source_node_set=source_node_set,
            )
        return repositories[repo]

    # Always create the primary repository node, even for an empty snapshot.
    _get_repository(fallback_repo)

    entities: list[DataPoint] = []
    skipped_facts = 0
    duplicate_facts = 0
    unmapped_kinds: Counter = Counter()
    seen_ids: set = set()

    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        model = KIND_TO_MODEL.get(kind) if isinstance(kind, str) else None

        if model is None:
            # The contract says additive vocabulary never bumps the format
            # version, so unknown kinds are expected over time; report what
            # was intentionally dropped once, per kind, instead of per fact.
            unmapped_kinds[kind if isinstance(kind, str) else repr(kind)] += 1
            continue
        if not isinstance(name, str) or not name:
            skipped_facts += 1
            logger.warning("Skipping fact with missing name: %s", fact)
            continue

        repo = _fact_repo(fact, fallback_repo)
        node_id = fact_node_id(repo, kind, name)
        if node_id in seen_ids:
            # Same-named facts of the same kind collapse into one node (see
            # fact_node_id). Keep the FIRST occurrence — the same rule the
            # storage-side deduplication applies — so the stored node content
            # (and its fact_hash) is deterministic across ingestions. Without
            # this, two duplicates with different content flip-flop the stored
            # hash and the fact reads as "updated" on every re-ingestion.
            duplicate_facts += 1
            continue
        seen_ids.add(node_id)
        props = fact.get("props")
        if not isinstance(props, dict):
            props = {}
        file_path = fact.get("file")
        line = fact.get("line")
        end_line = fact.get("end_line")
        enola_id = fact.get("id")

        fields: dict[str, Any] = {
            "id": node_id,
            "name": name,
            "kind": kind,
            "file_path": file_path if isinstance(file_path, str) else None,
            "line": line if isinstance(line, int) and not isinstance(line, bool) else None,
            "end_line": end_line
            if isinstance(end_line, int) and not isinstance(end_line, bool)
            else None,
            "repo": repo,
            "enola_id": enola_id if is_enola_id(enola_id) else None,
            # Insights carry prose from the explainer; use it verbatim instead
            # of the generic "kind: k=v, ..." property summary.
            "description": props.get("description")
            if kind == "insight" and isinstance(props.get("description"), str)
            else _describe_fact(kind, props),
            "fact_properties": props,
            "part_of": _get_repository(repo),
        }
        if model is CodeSymbol:
            fields["symbol_kind"] = props.get("symbol_kind")

        # node_set only enters the hash input when the call is tagged, so an
        # untagged repo's fact_hash stays byte-identical to before node_set
        # existed; a tagged repo's hash changes when its tag changes.
        hash_fields = {**fields, "node_set": node_set_hash_value} if node_set_hash_value else fields
        fields["fact_hash"] = _fact_content_hash(hash_fields)

        if node_sets is not None:
            fields["belongs_to_set"] = node_sets
            fields["source_node_set"] = source_node_set

        try:
            entities.append(model(**fields))
        except ValidationError:
            skipped_facts += 1
            logger.warning("Skipping fact with invalid field types: %s", fact)

    if skipped_facts:
        logger.warning("Skipped %d fact(s) that could not be mapped to DataPoints.", skipped_facts)
    if unmapped_kinds:
        logger.warning(
            "Dropped %d fact(s) of kind(s) cognee has no graph model for: %s",
            sum(unmapped_kinds.values()),
            dict(sorted(unmapped_kinds.items())),
        )
    if duplicate_facts:
        logger.info("Collapsed %d duplicate fact(s) into existing node ids.", duplicate_facts)

    return list(repositories.values()) + entities


def _fact_content_hash(fields: dict[str, Any]) -> str:
    """Fingerprint of a fact's derived fields, for delta writes on re-ingestion.

    Covers exactly what map_facts_to_data_points derives from the fact; the id
    and the part_of reference are excluded (both are functions of repo/kind/
    name, which are covered).
    """
    hashed_fields = {
        key: value for key, value in fields.items() if key not in ("id", "part_of", "fact_hash")
    }
    canonical = json.dumps(hashed_fields, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _short_target_names(name: str) -> set:
    """Unqualified forms of a fact name, e.g. 'app/db.Database' -> {'Database', 'db.Database'}.

    enola may reference a relation target by its bare name (e.g. an
    ``instantiates`` relation targeting ``Database`` while the symbol fact is
    named ``app/db.Database``), so edge resolution needs a suffix index.
    """
    forms = set()
    for form in (name.rsplit(".", 1)[-1], name.rsplit("/", 1)[-1]):
        if form and form != name:
            forms.add(form)
    return forms


def build_code_graph_edges(
    facts: list[dict],
    repo_path: str | Path | None = None,
) -> tuple[list[tuple], int]:
    """Resolve typed relations between facts into explicit graph edge tuples.

    A relation carrying the writer's ``target_id`` (enola >= 0.4.10) resolves
    to that fact directly — it is enola's own unambiguous answer. Otherwise
    the target is matched by fact name within the same snapshot; on ambiguity
    same-repo targets are preferred, then the relation is skipped rather than
    bound to an arbitrary fact. repo_path must be the same value given to
    map_facts_to_data_points so that edge endpoint ids match the node ids.
    Returns (edges, skipped_count).
    """
    fallback_repo = _resolve_fallback_repo(facts, repo_path)

    valid_facts = []
    name_index: dict[str, set] = {}
    short_name_index: dict[str, set] = {}
    enola_id_index: dict[str, tuple[str, str, str]] = {}
    fact_index: dict[tuple[str, str, str], dict] = {}
    module_index: dict[tuple[str, str], dict] = {}
    module_path_repos: dict[str, set[str]] = {}
    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if not _is_mappable_fact(kind, name):
            continue
        repo = _fact_repo(fact, fallback_repo)
        valid_facts.append((fact, repo))
        name_index.setdefault(name, set()).add((repo, kind))
        for short_form in _short_target_names(name):
            short_name_index.setdefault(short_form, set()).add((repo, kind, name))
        fact_id = fact.get("id")
        if is_enola_id(fact_id):
            # Records sharing an id share (repo, kind, name, file), hence the node.
            enola_id_index.setdefault(fact_id, (repo, kind, name))
        fact_index.setdefault((repo, kind, name), fact)
        if kind == "module":
            module_index.setdefault((repo, name), fact)
            props = fact.get("props")
            module_path = props.get("modulePath") if isinstance(props, dict) else None
            if isinstance(module_path, str) and module_path:
                module_path_repos.setdefault(module_path, set()).add(repo)

    edges: list[tuple] = []
    seen_edges = set()
    skipped = 0

    def _resolve_target(
        target_name: str,
        source_repo: str,
        allowed_repos: set[str] | None = None,
    ) -> tuple[str, str, str] | None:
        candidates = {(repo, kind, target_name) for repo, kind in name_index.get(target_name, ())}
        if not candidates:
            # No fact carries this exact name; fall back to unambiguous
            # suffix matches so bare-name targets (e.g. 'Database' for
            # 'app/db.Database') still resolve.
            candidates = set(short_name_index.get(target_name, ()))
        if allowed_repos is not None:
            candidates = {candidate for candidate in candidates if candidate[0] in allowed_repos}
        same_repo = {candidate for candidate in candidates if candidate[0] == source_repo}
        if len(same_repo) == 1:
            return next(iter(same_repo))
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    def _add_edge(
        source: tuple[str, str, str],
        target: tuple[str, str, str],
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
            target_id = relation_target_id(relation)
            target = enola_id_index.get(target_id) if target_id else None
            if target is None:
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
    modules_by_name: dict[str, list[tuple[str, dict]]] = {}
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
            target_module: tuple[str, dict] | None = None
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
    repo_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    timeout: float = 600.0,
    node_set: list[str] | None = None,
) -> list[DataPoint]:
    """Run enola on repo_path (or reuse an existing snapshot) and return DataPoints.

    The returned list composes with the add_data_points task downstream. Typed
    relations are persisted separately by add_code_graph_edges, which re-reads
    the same snapshot after the nodes exist in the graph. ``node_set`` tags
    every returned node (see ``map_facts_to_data_points``); a snapshot whose id
    is unchanged is still reloaded in full when node_set changed, so a re-tag
    is never skipped as "already loaded".
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
            "enola snapshot provenance: version=%s format_version=%s snapshot_id=%s facts=%s",
            receipt.get("enola_version"),
            receipt.get("format_version"),
            receipt.get("snapshot_id"),
            receipt.get("fact_count"),
        )

    snapshot_id = snapshot_identity(snapshot_dir, receipt)
    if snapshot_id is not None:
        fallback_repo = _resolve_fallback_repo(facts, repo_path)
        try:
            stored_id, stored_node_set = await _stored_repository_state(fallback_repo)
        except Exception as error:
            # The skip check is an optimization; never let it break ingestion.
            logger.warning(
                "Could not read the stored snapshot id (%s); loading fully.", error, exc_info=True
            )
            stored_id, stored_node_set = None, None
        current_node_set = sorted(node_set) if node_set else None
        if stored_id == snapshot_id and (
            (sorted(stored_node_set) if stored_node_set else None) == current_node_set
        ):
            logger.info(
                "Code graph for '%s' already matches snapshot %s (node_set unchanged); "
                "skipping load.",
                fallback_repo,
                snapshot_id,
            )
            return []

    data_points = map_facts_to_data_points(facts, repo_path=repo_path, node_set=node_set)
    logger.info("Mapped %d enola fact(s) to %d data point(s).", len(facts), len(data_points))
    return data_points


def _read_repo_node_property(node: Any, key: str) -> Any:
    """One property off a ``get_node()`` result.

    ``get_node`` returns either the properties flattened at the top level, or
    nested inside a ``properties`` key that may itself be JSON text (adapter
    dependent) -- both shapes are covered so a caller reads a property without
    caring which adapter answered.
    """
    if not isinstance(node, dict):
        return None
    value = node.get(key)
    if value is not None:
        return value
    properties = node.get("properties")
    if isinstance(properties, str):
        try:
            properties = json.loads(properties)
        except json.JSONDecodeError:
            return None
    if isinstance(properties, dict):
        return properties.get(key)
    return None


async def _stored_repository_state(repo: str) -> tuple[str | None, list[str] | None]:
    """(last_snapshot_id, last_node_set) recorded on the repository node by the
    last full load.

    Both live on the CodeRepository node in the graph itself — not in the
    relational metastore — because this pipeline persists no Data row to key
    relational state on (the payload is a repo path), and because a marker
    stored with the graph can never outlive it: forget(memory_only=True),
    prune, and even manual deletion of the graph database files all take the
    marker down with the data it describes. One read serves both the
    snapshot-skip check and the node_set comparison; this read is a separate,
    best-effort optimization from the one add_code_graph_data_points takes
    right before writing, which is the one that actually carries
    last_node_set forward (see that function's docstring) -- a failure here
    only costs a full reload, never a wrong marker.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    graph_engine = await get_graph_engine()
    node = await graph_engine.get_node(str(fact_node_id(repo, "repository", repo)))
    stored_id = _read_repo_node_property(node, "last_snapshot_id")
    stored_node_set = _read_repo_node_property(node, "last_node_set")
    return (
        stored_id if isinstance(stored_id, str) and stored_id else None,
        stored_node_set if isinstance(stored_node_set, list) else None,
    )


def _snapshot_repos(facts: list[dict], fallback_repo: str) -> set:
    """Every repo this snapshot covers (multi-repo snapshots have several)."""
    repos = {fallback_repo}
    for fact in facts:
        if _is_mappable_fact(fact.get("kind"), fact.get("name")):
            repos.add(_fact_repo(fact, fallback_repo))
    return repos


def _current_code_node_ids(facts: list[dict], fallback_repo: str) -> set:
    """String node ids the snapshot derives: every mappable fact + repository nodes."""
    ids = set()
    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if _is_mappable_fact(kind, name):
            ids.add(str(fact_node_id(_fact_repo(fact, fallback_repo), kind, name)))
    for repo in _snapshot_repos(facts, fallback_repo):
        ids.add(str(fact_node_id(repo, "repository", repo)))
    return ids


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


class _CodeGraphLoadState(list):
    """The data_points passthrough between the load and edges tasks.

    Also carries the pre-write graph state and the node delta computed by
    add_code_graph_data_points, so the edge diff and the stale sweep in
    add_code_graph_edges reuse the same single graph read.
    """

    existing_edge_keys: set | None = None
    existing_nodes: list | None = None
    node_delta: dict | None = None


_DELTA_SAMPLE_LIMIT = 20


def _delta_samples(names: list[str]) -> list[str]:
    return sorted(names)[:_DELTA_SAMPLE_LIMIT]


async def add_code_graph_data_points(
    data_points: list[DataPoint],
    ctx: Optional["PipelineContext"] = None,
    graph_only: bool = True,
) -> list[DataPoint]:
    """Store code graph nodes while allowing a repository path payload.

    Delta writes: the graph is read once before writing and only facts whose
    fact_hash is new or changed are stored; unchanged facts are not touched.
    The pre-read state rides on the returned list so add_code_graph_edges can
    diff edges and sweep without reading the graph again.

    This is also the one place that carries the CodeRepository node's
    ``last_node_set`` marker forward. This write is a full rewrite of the
    node on Ladybug/postgres_demo, so whatever value the DataPoint carries
    replaces the stored one outright; the marker is set here, from the read
    this function already takes, rather than from a value computed earlier
    in extract_code_graph -- that earlier read is a separate, independently
    fallible optimization (the snapshot-skip check), and if it failed or
    went stale between there and here, trusting it would wipe the real
    previous tag to ``None``. This read either succeeds (and the marker
    carries forward correctly) or raises (and nothing gets written at all),
    so there is no path that writes a wrong marker.

    A custom-pipeline payload may be any value, but the storage rollback ledger
    requires a persisted data item id. Preserve the full context when one is
    available and otherwise store without ledger provenance. graph_only keeps
    the deterministic default free of embedding calls; set it to False to also
    build vector indexes for completion-based search types.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
    from cognee.tasks.storage.add_data_points import add_data_points

    if not data_points:
        # extract_code_graph skipped an unchanged snapshot.
        return data_points

    graph_engine = await get_graph_engine()
    existing_nodes, existing_edges = await graph_engine.get_graph_data()
    existing_hashes: dict[str, Any] = {
        str(node_id): properties.get("fact_hash")
        for node_id, properties in existing_nodes
        if isinstance(properties, dict)
    }
    existing_repo_last_node_set: dict[str, Any] = {
        str(node_id): properties.get("last_node_set")
        for node_id, properties in existing_nodes
        if isinstance(properties, dict) and properties.get("type") == "CodeRepository"
    }

    to_write: list[DataPoint] = []
    added: list[str] = []
    updated: list[str] = []
    unchanged = 0
    for point in data_points:
        if isinstance(point, CodeRepository):
            # Repository nodes are always rewritten (they carry the snapshot
            # stamp) and are not counted as content changes. last_node_set
            # carries forward from this read (see docstring above).
            stored_last_node_set = existing_repo_last_node_set.get(str(point.id))
            point.last_node_set = (
                stored_last_node_set if isinstance(stored_last_node_set, list) else None
            )
            to_write.append(point)
            continue
        point_id = str(getattr(point, "id", point))
        fact_hash = getattr(point, "fact_hash", None)
        if point_id not in existing_hashes:
            to_write.append(point)
            added.append(str(getattr(point, "name", point_id)))
        elif fact_hash is None or existing_hashes[point_id] != fact_hash:
            to_write.append(point)
            updated.append(str(getattr(point, "name", point_id)))
        else:
            unchanged += 1

    logger.info(
        "Code graph node delta: %d added, %d updated, %d unchanged.",
        len(added),
        len(updated),
        unchanged,
    )

    storage_ctx = ctx if _pipeline_data_id(ctx) is not None else None
    try:
        if to_write:
            await add_data_points(to_write, ctx=storage_ctx, graph_only=graph_only)
    finally:
        # Storage can fail after a partial graph write. Invalidate even on an
        # exception so no pre-write snapshot survives that ambiguous outcome.
        _invalidate_code_graph_snapshot(ctx)

    state = _CodeGraphLoadState(data_points)
    state.existing_nodes = existing_nodes
    state.existing_edge_keys = {
        (str(source), str(target), relationship)
        for source, target, relationship, _properties in existing_edges
    }
    state.node_delta = {
        "nodes_added": len(added),
        "nodes_updated": len(updated),
        "nodes_unchanged": unchanged,
        "samples_added": _delta_samples(added),
        "samples_updated": _delta_samples(updated),
    }
    return state


async def add_code_graph_edges(
    data_points: list[DataPoint],
    repo_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    ctx: Optional["PipelineContext"] = None,
    node_set: list[str] | None = None,
) -> list[DataPoint]:
    """Insert typed relation edges (calls/imports/...) after add_data_points ran.

    Relation names are dynamic, so they cannot be expressed as DataPoint field
    references; instead they are written directly through the graph engine,
    following the extract_dlt_fk_edges precedent. Afterwards, stale nodes and
    edges from earlier ingestions of the same repos are swept, a node_set
    re-tag (if any) is detagged off the nodes that no longer carry it, and the
    snapshot identity is stamped on the repository node so the next unchanged
    ingestion can skip entirely. Passthrough: returns data_points unchanged.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    if not data_points:
        # extract_code_graph skipped an unchanged snapshot; nothing to add,
        # sweep, or stamp.
        return data_points

    if snapshot_dir is None:
        if repo_path is None:
            raise ValueError("add_code_graph_edges requires repo_path or snapshot_dir.")
        snapshot_dir = Path(repo_path) / ".enola"

    facts, receipt = parse_enola_snapshot(snapshot_dir)
    edges, skipped = build_code_graph_edges(facts, repo_path=repo_path)
    logger.info("Resolved %d code graph edge(s), skipped %d.", len(edges), skipped)

    try:
        graph_engine = await get_graph_engine()

        # Pre-write graph state: reuse the read add_code_graph_data_points
        # already did (it rides on the passthrough list); direct callers pay
        # one read here instead.
        existing_nodes = getattr(data_points, "existing_nodes", None)
        existing_edge_keys = getattr(data_points, "existing_edge_keys", None)
        if existing_nodes is None or existing_edge_keys is None:
            existing_nodes, existing_edges = await graph_engine.get_graph_data()
            existing_edge_keys = {
                (str(source), str(target), relationship)
                for source, target, relationship, _properties in existing_edges
            }

        # Delta writes: only edges the graph does not already have.
        new_edges = [
            edge
            for edge in edges
            if (str(edge[0]), str(edge[1]), edge[2]) not in existing_edge_keys
        ]
        logger.info(
            "Code graph edge delta: %d new, %d already present.",
            len(new_edges),
            len(edges) - len(new_edges),
        )
        if new_edges:
            await graph_engine.add_edges(new_edges)

            # Register the edges added by this run in the relational rollback
            # ledger (when a pipeline context with a persisted data item is
            # available) so pipeline rollback can clean them up. Custom
            # pipelines may use arbitrary payloads, such as the repository
            # path used by the code graph example.
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
                    new_edges,
                    tenant_id=ctx.user.tenant_id,
                    user_id=ctx.user.id,
                    dataset_id=ctx.dataset.id,
                    data_id=data_id,
                    pipeline_run_id=ctx.pipeline_run_id,
                )

        # Insert-then-sweep: with the current snapshot fully merged, remove
        # what previous ingestions derived that this snapshot no longer does.
        # Runs even with zero edges — a shrunken repo still needs its sweep.
        nodes_removed, edges_removed, samples_removed = await _sweep_stale_code_graph(
            graph_engine, facts, edges, repo_path, existing_nodes, existing_edge_keys
        )

        # Retag: strip node_set names this repo carried before but not anymore.
        # A no-op (zero graph calls) when node_set is unchanged, including the
        # untagged case, so the per-file CODE route and the legacy
        # remember(content_type="code") path (both always pass node_set=None)
        # never pay for this.
        detagged = await _detag_stale_code_graph(
            graph_engine, facts, repo_path, existing_nodes, existing_edge_keys, node_set
        )

        snapshot_id = snapshot_identity(snapshot_dir, receipt)
        node_delta = getattr(data_points, "node_delta", None) or {}
        delta = {
            **node_delta,
            "edges_added": len(new_edges),
            "edges_removed": edges_removed,
            "nodes_removed": nodes_removed,
            "samples_removed": samples_removed,
            "node_set_detagged": detagged,
            "snapshot_id": snapshot_id,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }

        # Stamp last: only a load that added, swept, detagged, and got here may
        # record its snapshot id, so a crashed run can never be skipped-past later.
        await _stamp_snapshot_identity(
            graph_engine, facts, repo_path, snapshot_id, delta, receipt=receipt, node_set=node_set
        )
    finally:
        # Direct edge writes, sweeps, and ledger writes may partially succeed.
        _invalidate_code_graph_snapshot(ctx)
    return data_points


def _code_graph_node_types() -> set:
    """Every graph node type this pipeline produces, CodeRepository included."""
    return {model.__name__ for model in KIND_TO_MODEL.values()} | {CodeRepository.__name__}


def _is_repo_code_node(properties: Any, snapshot_repos: set, code_types: set) -> bool:
    """Whether a stored node's properties are a code-graph node of one of these repos.

    Shared by the stale-node sweep and the stale node_set detag, so both agree
    on exactly which nodes belong to a snapshot's repos.
    """
    if not isinstance(properties, dict):
        return False
    node_type = properties.get("type")
    if node_type not in code_types:
        return False
    # Repository nodes carry their repo in name; entities in the repo field.
    repo = properties.get("name") if node_type == "CodeRepository" else properties.get("repo")
    return repo in snapshot_repos


async def _sweep_stale_code_graph(
    graph_engine,
    facts: list[dict],
    current_edges: list[tuple],
    repo_path: str | Path | None,
    existing_nodes: list[tuple],
    existing_edge_keys: set,
) -> tuple[int, int, list[str]]:
    """Remove code graph nodes/edges no longer derivable from the snapshot.

    existing_nodes/existing_edge_keys are the pre-write graph state (the same
    read the node-delta computation used). Only nodes of code-graph types
    belonging to repos covered by this snapshot are considered; other
    datasets' content and other repos in the same graph are untouched. Edges
    are swept only between surviving code nodes, so edges to non-code nodes
    (e.g. belongs_to_set -> NodeSet) always survive. Returns
    (nodes_removed, edges_removed, removed_name_samples).
    """
    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity

    fallback_repo = _resolve_fallback_repo(facts, repo_path)
    snapshot_repos = _snapshot_repos(facts, fallback_repo)
    current_ids = _current_code_node_ids(facts, fallback_repo)

    code_types = _code_graph_node_types()
    stale_node_ids = []
    stale_node_names = []
    for node_id, properties in existing_nodes:
        node_id = str(node_id)
        if node_id in current_ids:
            continue
        if not _is_repo_code_node(properties, snapshot_repos, code_types):
            continue
        stale_node_ids.append(node_id)
        stale_node_names.append(str(properties.get("name") or node_id))

    if stale_node_ids:
        for start in range(0, len(stale_node_ids), _SWEEP_CHUNK_SIZE):
            await graph_engine.delete_nodes(stale_node_ids[start : start + _SWEEP_CHUNK_SIZE])
        logger.info("Swept %d stale code graph node(s).", len(stale_node_ids))

    expected_edge_keys = {
        (str(source_id), str(target_id), relationship_name)
        for source_id, target_id, relationship_name, _properties in current_edges
    }
    # Structural containment edges written by add_data_points from the
    # DataPoint part_of field.
    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if not _is_mappable_fact(kind, name):
            continue
        repo = _fact_repo(fact, fallback_repo)
        expected_edge_keys.add(
            (
                str(fact_node_id(repo, kind, name)),
                str(fact_node_id(repo, "repository", repo)),
                "part_of",
            )
        )

    stale_edges = [
        EdgeIdentity(source_id=source, target_id=target, relationship_name=relationship)
        for source, target, relationship in existing_edge_keys
        if source in current_ids
        and target in current_ids
        and (source, target, relationship) not in expected_edge_keys
    ]
    if stale_edges:
        for start in range(0, len(stale_edges), _SWEEP_CHUNK_SIZE):
            await graph_engine.delete_edge_triples(stale_edges[start : start + _SWEEP_CHUNK_SIZE])
        logger.info("Swept %d stale code graph edge(s).", len(stale_edges))

    return len(stale_node_ids), len(stale_edges), _delta_samples(stale_node_names)


async def _detag_stale_code_graph(
    graph_engine,
    facts: list[dict],
    repo_path: str | Path | None,
    existing_nodes: list[tuple],
    existing_edge_keys: set,
    node_set: list[str] | None,
) -> list[str]:
    """Strip node_set tag names this repo no longer carries after a re-tag.

    The repository's ``last_node_set`` marker (stamped only by a fully
    completed load) is one signal, but not the only one: a run that writes a
    tag and then crashes before its own stamp leaves that tag on the
    belongs_to_set property and edges without ever advancing the marker, so
    a marker-only comparison can never see it and the tag is stranded
    forever. The set of names to strip is instead the union of:
      - the marker (last_node_set on the repository node, if any),
      - the belongs_to_set property actually stored on this repo's surviving
        code nodes in this pre-write read, and
      - the NodeSet names resolved from any belongs_to_set edge these nodes
        currently hold (via the NodeSet nodes' own ``name`` property, read
        from this same existing_nodes snapshot),
    minus the current ``node_set`` -- but the two writes this produces are
    stale by DIFFERENT comparisons, because they fail in opposite directions
    otherwise:

    - Edges are compared by NodeSet id (``generate_node_id``, which
      lowercases, turns spaces into underscores and strips apostrophes), so
      "Team-A" and "team-a" (or "team a" and "team_a") name the SAME
      NodeSet under different spellings. A raw-string comparison here would
      treat a same-id retag as a real removal and delete the edge the
      current write just added.
    - The property is compared by EXACT string instead, because
      ``remove_belongs_to_set_tags`` filters by literal string membership on
      every backend (Neo4j, Ladybug, postgres_demo, Turso). An id-based
      comparison here would never even propose stripping "Team-A" once
      "team-a" is the current tag (same id, so not "stale"), and Neo4j's
      node rewrite UNIONS ``belongs_to_set`` instead of replacing it -- so a
      respelling-only retag would keep both spellings on the property
      forever. Passing the id-equivalent old spelling to
      remove_belongs_to_set_tags is safe: it matches by literal string, so
      "Team-A" can never remove "team-a".

    Names dropped are removed from every surviving code node of this repo
    (post-sweep survivors only -- a node about to be deleted needs no
    detagging): the belongs_to_set edges to the retired NodeSet(s) via
    delete_edge_triples (needed on every backend, since a node rewrite only
    adds edges, never removes them), and the belongs_to_set property array
    via remove_belongs_to_set_tags (writes nothing on backends whose node
    rewrite already replaced the property (one scoped read); load-bearing on
    Neo4j, whose rewrite unions belongs_to_set instead of replacing it).
    Returns the union of names touched by either write, or an empty list
    when there was nothing to strip.
    """
    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity

    fallback_repo = _resolve_fallback_repo(facts, repo_path)
    snapshot_repos = _snapshot_repos(facts, fallback_repo)
    current_ids = _current_code_node_ids(facts, fallback_repo)
    code_types = _code_graph_node_types()

    code_node_ids = [
        str(node_id)
        for node_id, properties in existing_nodes
        if str(node_id) in current_ids
        and _is_repo_code_node(properties, snapshot_repos, code_types)
    ]
    if not code_node_ids:
        return []
    code_node_id_set = set(code_node_ids)

    stored_node_set: list | None = None
    property_tag_names: set = set()
    nodeset_id_to_name: dict = {}
    for node_id, properties in existing_nodes:
        if not isinstance(properties, dict):
            continue
        node_type = properties.get("type")
        if node_type == "NodeSet":
            name = properties.get("name")
            if isinstance(name, str):
                nodeset_id_to_name[str(node_id)] = name
            continue
        if node_type == "CodeRepository" and properties.get("name") in snapshot_repos:
            candidate = properties.get("last_node_set")
            if isinstance(candidate, list):
                stored_node_set = candidate
        if str(node_id) in code_node_id_set:
            belongs_to_set = properties.get("belongs_to_set")
            if isinstance(belongs_to_set, list):
                property_tag_names.update(name for name in belongs_to_set if isinstance(name, str))

    edge_tag_names: set = set()
    for source, target, relationship in existing_edge_keys or ():
        if relationship != "belongs_to_set" or source not in code_node_id_set:
            continue
        name = nodeset_id_to_name.get(target)
        if name:
            edge_tag_names.add(name)

    candidate_names = set(stored_node_set or []) | property_tag_names | edge_tag_names

    # Edges: stale by NORMALIZED id -- never delete an edge whose id matches
    # a current name (a same-id retag is a respelling, not a removal).
    current_name_ids = {str(generate_node_id(f"NodeSet:{name}")) for name in (node_set or [])}
    stale_edge_names = sorted(
        name
        for name in candidate_names
        if str(generate_node_id(f"NodeSet:{name}")) not in current_name_ids
    )

    # Property: stale by EXACT STRING -- an old spelling that shares its
    # NodeSet id with the current tag must still be stripped from the
    # property text itself (see docstring for why this is safe and why the
    # id-based comparison above cannot be reused here).
    stale_property_names = sorted(property_tag_names - set(node_set or []))

    if not stale_edge_names and not stale_property_names:
        return []

    if stale_edge_names:
        stale_edges = [
            EdgeIdentity(
                source_id=node_id,
                target_id=str(generate_node_id(f"NodeSet:{name}")),
                relationship_name="belongs_to_set",
            )
            for node_id in code_node_ids
            for name in stale_edge_names
        ]
        for start in range(0, len(stale_edges), _SWEEP_CHUNK_SIZE):
            await graph_engine.delete_edge_triples(stale_edges[start : start + _SWEEP_CHUNK_SIZE])

    if stale_property_names:
        for start in range(0, len(code_node_ids), _SWEEP_CHUNK_SIZE):
            await graph_engine.remove_belongs_to_set_tags(
                stale_property_names, node_ids=code_node_ids[start : start + _SWEEP_CHUNK_SIZE]
            )

    removed = sorted(set(stale_edge_names) | set(stale_property_names))
    logger.info(
        "Removed stale node_set tag(s) %s from %d code graph node(s).",
        removed,
        len(code_node_ids),
    )
    return removed


def receipt_projection(receipt: dict | None) -> dict | None:
    """The receipt.json fields kept on the repository node (see _RECEIPT_PROJECTION_KEYS)."""
    if not isinstance(receipt, dict):
        return None
    projection = {key: receipt[key] for key in _RECEIPT_PROJECTION_KEYS if key in receipt}
    return projection or None


async def _stamp_snapshot_identity(
    graph_engine,
    facts: list[dict],
    repo_path: str | Path | None,
    snapshot_id: str | None,
    delta: dict | None = None,
    receipt: dict | None = None,
    node_set: list[str] | None = None,
) -> None:
    """Record the loaded snapshot's identity, node_set tag, delta and receipt
    on the repository nodes.

    This is a full rewrite of the CodeRepository node (``add_nodes``), so it
    must carry belongs_to_set/source_node_set/last_node_set forward itself --
    a rewrite that omitted them would erase the tag add_code_graph_data_points
    just wrote (Ladybug/postgres_demo overwrite node properties wholesale;
    Neo4j unions belongs_to_set, so re-sending the current tag here is a
    no-op there, not a duplicate). This write goes through ``add_nodes``
    directly, not ``get_graph_from_model``, so ``belongs_to_set`` must already
    be the stored shape (tag names) rather than ``NodeSet`` DataPoints --
    passing DataPoints here would serialize full NodeSet objects onto the
    property instead of names.
    """
    if snapshot_id is None:
        return
    fallback_repo = _resolve_fallback_repo(facts, repo_path)
    last_receipt = receipt_projection(receipt)
    source_node_set = ", ".join(node_set) if node_set else None
    last_node_set = sorted(node_set) if node_set else None
    repositories = [
        CodeRepository(
            id=fact_node_id(repo, "repository", repo),
            name=repo,
            path=str(repo_path) if repo_path and repo == fallback_repo else repo,
            last_snapshot_id=snapshot_id,
            last_delta=delta,
            last_receipt=last_receipt,
            last_node_set=last_node_set,
            belongs_to_set=last_node_set,
            source_node_set=source_node_set,
        )
        for repo in sorted(_snapshot_repos(facts, fallback_repo))
    ]
    await graph_engine.add_nodes(repositories)


def get_code_graph_tasks(
    repo_path: str | Path,
    snapshot_dir: str | Path | None = None,
    timeout: float = 600.0,
    index_vectors: bool = False,
) -> list[Task]:
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
