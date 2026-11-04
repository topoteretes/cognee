"""Map enola snapshot facts to cognee DataPoints and typed graph edges."""

import hashlib
import json
import posixpath
from collections import Counter
from collections.abc import AsyncIterator, Callable, Generator, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional
from uuid import NAMESPACE_OID, UUID, uuid5

from pydantic import ValidationError

from cognee.infrastructure.databases.graph.graph_db_interface import (
    DEFAULT_GRAPH_STREAM_BATCH_SIZE,
)
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines.tasks.task import Task
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.enola import (
    EnolaSnapshot,
    ParsedSnapshot,
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
    CodeFlow,
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
    "flow": CodeFlow,
    "intent": CodeIntent,
    "extraction": CodeExtractionAccount,
    "association": CodeAssociation,
    "lint": CodeLintFinding,
}

# The prop holding the prose a synthesized kind is described by: an insight's
# explainer text, a flow's summary.
_PROSE_PROPS = {"insight": "description", "flow": "summary"}

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

# Version of how cognee maps an enola snapshot onto graph nodes and edges.
# Bump it whenever that mapping changes, so a repository whose snapshot did not
# change is still loaded again by the new mapping instead of being skipped.
# 1: doc fields, CodeFlow nodes and declaration-view resolution (SDK-1119).
CODE_GRAPH_MAPPING_VERSION = 1


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


def _resolve_fallback_repo(facts: Iterable[dict], repo_path: str | Path | None) -> str:
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
    repo_scope: str | None = None,
    node_set: list[str] | None = None,
) -> list[DataPoint]:
    """Map parsed enola facts to DataPoints, one CodeRepository per repo first.

    The list form of ``iter_fact_data_points``, over facts already in memory.
    """
    points = list(
        iter_fact_data_points(
            facts,
            repo_path=repo_path,
            repo_scope=repo_scope,
            node_set=node_set,
            fallback_repo=_resolve_fallback_repo(facts, repo_path),
        )
    )
    return [point for point in points if isinstance(point, CodeRepository)] + [
        point for point in points if not isinstance(point, CodeRepository)
    ]


def iter_fact_data_points(
    facts: Iterable[dict],
    repo_path: str | Path | None,
    repo_scope: str | None,
    node_set: list[str] | None,
    fallback_repo: str,
) -> Iterator[DataPoint]:
    """Map enola facts to DataPoints one at a time, each CodeRepository before its facts.

    The primary repository's node comes first, even for an empty snapshot;
    another repository's node comes just before its first fact. A fact whose
    node id an earlier fact already took is dropped, so a node is yielded once.
    ``fallback_repo`` is ``_resolve_fallback_repo`` of the same facts.

    repo_scope narrows a multi-repo snapshot to one member: only that
    repository's facts are mapped, and repo_path is that member's directory.

    ``node_set`` (the caller's add()/remember() tag, read off the repo
    manifest's external_metadata by extract_code_repo_graph) tags every
    returned node -- CodeRepository included -- with the same
    belongs_to_set/source_node_set a document gets from classify_documents.
    An untagged call (node_set=None, the per-file CODE route and the legacy
    remember(content_type="code") path) produces byte-identical fact_hash
    values to before this parameter existed.
    """
    primary_repo = repo_scope or fallback_repo

    node_sets = [NodeSet(name=name) for name in node_set] if node_set else None
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
                path=str(repo_path) if repo_path and repo == primary_repo else repo,
                belongs_to_set=node_sets,
                source_node_set=source_node_set,
            )
        return repositories[repo]

    # Always create the primary repository node, even for an empty snapshot.
    yield _get_repository(primary_repo)

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
        if repo_scope is not None and repo != repo_scope:
            continue
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
        doc = props.get("doc")
        if kind == "insight" or not isinstance(doc, str) or not doc.strip():
            doc = None
        prose = props.get(_PROSE_PROPS[kind]) if kind in _PROSE_PROPS else doc

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
            # of the generic "kind: k=v, ..." property summary. So do flows
            # (their summary) and documented facts (the author's doc comment).
            "description": prose if isinstance(prose, str) else _describe_fact(kind, props),
            "fact_properties": props,
        }
        if repo not in repositories:
            yield _get_repository(repo)
        fields["part_of"] = _get_repository(repo)
        if model is CodeSymbol:
            fields["symbol_kind"] = props.get("symbol_kind")
        if doc is not None:
            # Only a documented fact gains the key, so an undocumented fact's
            # fact_hash stays byte-identical to before docs were read.
            fields["doc"] = doc

        # node_set only enters the hash input when the call is tagged, so an
        # untagged repo's fact_hash stays byte-identical to before node_set
        # existed; a tagged repo's hash changes when its tag changes.
        hash_fields = {**fields, "node_set": node_set_hash_value} if node_set_hash_value else fields
        fields["fact_hash"] = _fact_content_hash(hash_fields)

        if node_sets is not None:
            fields["belongs_to_set"] = node_sets
            fields["source_node_set"] = source_node_set

        try:
            point = model(**fields)
        except ValidationError:
            point = None
            skipped_facts += 1
            logger.warning("Skipping fact with invalid field types: %s", fact)
        if point is not None:
            yield point

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


def _add_client_route_edges(
    client_routes: list[tuple[str, str, dict]],
    enola_id_index: dict[str, tuple[str, str, str]],
    fact_index: dict[tuple[str, str, str], "_FactRef"],
    add_edge: Callable[..., None],
) -> int:
    """Edges from a client route's caller and matched_routes props; returns skips.

    A client route names the symbol that makes the call (caller) and, in a
    multi-repo snapshot, the server routes it reaches (matched_routes). Both
    are props rather than relations because a call and the route it reaches
    usually share a name; each carries the writer's fact id, which resolves
    first. The name fallback stays inside the repo the prop names.
    """

    def _resolve_ref(fact_id: Any, repo: Any, kind: str, ref_name: Any):
        if is_enola_id(fact_id) and fact_id in enola_id_index:
            return enola_id_index[fact_id]
        if isinstance(repo, str) and isinstance(ref_name, str):
            key = (repo, kind, ref_name)
            if key in fact_index:
                return key
        return None

    skipped = 0
    for route_name, source_repo, props in client_routes:
        route = (source_repo, "route", route_name)

        caller = props.get("caller")
        if caller is not None or props.get("caller_id") is not None:
            resolved = _resolve_ref(props.get("caller_id"), source_repo, "symbol", caller)
            if resolved is None:
                skipped += 1
            else:
                add_edge(resolved, route, "makes_request")

        matches = props.get("matched_routes")
        for match in matches if isinstance(matches, list) else []:
            if not isinstance(match, dict):
                skipped += 1
                continue
            resolved = _resolve_ref(match.get("id"), match.get("repo"), "route", match.get("name"))
            if resolved is None:
                skipped += 1
                continue
            if resolved == route:
                # A call its own repo serves under the same path collapses
                # onto the served route's node; a self-loop says nothing.
                continue
            confidence = match.get("confidence")
            add_edge(
                route,
                resolved,
                "reaches_route",
                {"confidence": confidence} if isinstance(confidence, str) else None,
            )
    return skipped


# Fact kinds a framework extractor derives from a declaration and names after
# it: an ORM model's storage (Eloquent, JPA, ActiveRecord) and a relation
# method's association. code_retriever resolves names the same way.
_DECLARATION_VIEW_KINDS = frozenset({"storage", "association"})


def _without_declaration_views(
    candidates: set[tuple[str, str, str]],
    fact_index: dict[tuple[str, str, str], "_FactRef"],
) -> set[tuple[str, str, str]]:
    """Name candidates minus the views of a symbol that is itself a candidate.

    A view is a storage or association fact carrying the symbol's own name,
    repo and file. Anything else sharing the name (a fact from another file or
    repo, a second same-named symbol) stays a rival candidate. When one repo
    declares the symbol name in several files, the file compared is that of the
    first such fact, the one its node keeps.
    """

    def _is_view(candidate: tuple[str, str, str]) -> bool:
        repo, kind, name = candidate
        symbol = (repo, "symbol", name)
        if kind not in _DECLARATION_VIEW_KINDS or symbol not in candidates:
            return False
        symbol_file = fact_index[symbol].file
        return (
            isinstance(symbol_file, str)
            and symbol_file != ""
            and fact_index[candidate].file == symbol_file
        )

    return {candidate for candidate in candidates if not _is_view(candidate)}


class _FactRef(NamedTuple):
    """What edge resolution reads off a fact besides its identity."""

    file: Any
    symbol_kind: Any


@dataclass
class _SnapshotScan:
    """One pass over a snapshot's facts: what edge resolution and the sweep read.

    ``node_repos`` maps every node id the snapshot derives (each mappable fact
    and each repository node) to the id of the repository node its ``part_of``
    edge points at (None for a repository node itself): membership is the
    snapshot's current node ids, the values its structural edges. The rest is
    the resolver's index, with no fact dict kept: ``fact_index`` holds a
    ``_FactRef`` per (repo, kind, name), the first fact's, and the three small
    passes the resolver makes after the relations read the compact lists.
    """

    fallback_repo: str
    repos: set[str] = field(default_factory=set)
    node_repos: dict[str, str | None] = field(default_factory=dict)
    name_index: dict[str, set] = field(default_factory=dict)
    short_name_index: dict[str, set] = field(default_factory=dict)
    enola_id_index: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    fact_index: dict[tuple[str, str, str], _FactRef] = field(default_factory=dict)
    module_index: dict[tuple[str, str], str] = field(default_factory=dict)
    module_path_repos: dict[str, set[str]] = field(default_factory=dict)
    # (route name, repo, props) of each client route, in snapshot order.
    client_routes: list[tuple[str, str, dict]] = field(default_factory=list)
    # (repo, file, import targets) of each dependency fact with a file.
    dependencies: list[tuple[str, str, list[str]]] = field(default_factory=list)
    # (repo, name) of each method or function symbol with a qualified name.
    method_symbols: list[tuple[str, str]] = field(default_factory=list)

    def scoped_node_ids(self, repo_scope: str) -> set[str]:
        """The node ids of one repository of the snapshot, its repository node included."""
        repo_node_id = str(fact_node_id(repo_scope, "repository", repo_scope))
        return {
            node_id for node_id, repo_id in self.node_repos.items() if repo_id == repo_node_id
        } | {repo_node_id}


def _scan_snapshot(facts: Iterable[dict], fallback_repo: str) -> _SnapshotScan:
    """Index a snapshot's facts for edge resolution and the sweep, in one pass."""
    scan = _SnapshotScan(fallback_repo=fallback_repo, repos={fallback_repo})
    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if not _is_mappable_fact(kind, name):
            continue
        repo = _fact_repo(fact, fallback_repo)
        if repo not in scan.repos:
            scan.repos.add(repo)
        scan.node_repos[str(fact_node_id(repo, kind, name))] = str(
            fact_node_id(repo, "repository", repo)
        )
        if kind != "flow":
            # No relation targets a flow, and a flow is named after its entry
            # symbol or route: indexing its name would make those names
            # ambiguous for every relation that resolves by name.
            scan.name_index.setdefault(name, set()).add((repo, kind))
            for short_form in _short_target_names(name):
                scan.short_name_index.setdefault(short_form, set()).add((repo, kind, name))
        fact_id = fact.get("id")
        if is_enola_id(fact_id):
            # Records sharing an id share (repo, kind, name, file), hence the node.
            scan.enola_id_index.setdefault(fact_id, (repo, kind, name))
        props = fact.get("props")
        if not isinstance(props, dict):
            props = None
        scan.fact_index.setdefault(
            (repo, kind, name),
            _FactRef(fact.get("file"), props.get("symbol_kind") if props else None),
        )
        if kind == "module":
            scan.module_index.setdefault((repo, name), name)
            module_path = props.get("modulePath") if props else None
            if isinstance(module_path, str) and module_path:
                scan.module_path_repos.setdefault(module_path, set()).add(repo)
        elif kind == "route" and props is not None and props.get("role") == "client":
            scan.client_routes.append((name, repo, props))
        elif kind == "dependency" and isinstance(fact.get("file"), str):
            imports = []
            for relation in fact.get("relations") or []:
                normalized = normalize_relation(relation)
                if normalized is not None and normalized[0] == "imports":
                    imports.append(normalized[1])
            scan.dependencies.append((repo, fact["file"], imports))
        elif (
            kind == "symbol"
            and "." in name
            and props is not None
            and props.get("symbol_kind") in {"method", "function"}
        ):
            scan.method_symbols.append((repo, name))
    for repo in scan.repos:
        scan.node_repos[str(fact_node_id(repo, "repository", repo))] = None
    return scan


def build_code_graph_edges(
    facts: list[dict],
    repo_path: str | Path | None = None,
) -> tuple[list[tuple], int]:
    """Every edge ``iter_code_graph_edges`` resolves from ``facts``, as one list.

    Returns (edges, skipped_count). repo_path must be the same value given to
    map_facts_to_data_points so that edge endpoint ids match the node ids.
    """
    scan = _scan_snapshot(facts, _resolve_fallback_repo(facts, repo_path))
    edges: list[tuple] = []
    skipped = _drain(iter_code_graph_edges(scan, facts, set()), edges.append)
    return edges, skipped


def _drain(edges: Generator[tuple, None, int], consume: Callable[[tuple], None]) -> int:
    """Hand every edge of a resolver generator to ``consume``; return its skipped count."""
    while True:
        try:
            consume(next(edges))
        except StopIteration as stop:
            return stop.value


def _resolved(edges: list[tuple]) -> Generator[tuple, None, int]:
    """Edges already resolved (a cluster's shared list), in the resolver's shape."""
    yield from edges
    return 0


def iter_code_graph_edges(
    scan: _SnapshotScan,
    facts: Iterable[dict],
    edge_keys: set[tuple[str, str, str]],
) -> Generator[tuple, None, int]:
    """Resolve typed relations between facts into graph edge tuples, one at a time.

    ``scan`` is ``_scan_snapshot`` of the same facts, and ``facts`` a fresh
    pass over them, read for their relations. Each edge is yielded once: its
    ``(source_id, target_id, relationship_name)`` key, as strings, goes into
    ``edge_keys`` as it resolves and a repeat is dropped, so the caller's
    sweep reads the same set. Returns the number of relations skipped.

    A relation carrying the writer's ``target_id`` (enola >= 0.4.10) resolves
    to that fact directly — it is enola's own unambiguous answer. Otherwise
    the target is matched by fact name within the same snapshot; on ambiguity
    same-repo targets are preferred, then the relation is skipped rather than
    bound to an arbitrary fact. A storage or association fact sharing a
    symbol's name, repo and file is a view of that symbol, not a rival: the
    name means the symbol.
    """
    fallback_repo = scan.fallback_repo
    name_index = scan.name_index
    short_name_index = scan.short_name_index
    enola_id_index = scan.enola_id_index
    fact_index = scan.fact_index
    module_index = scan.module_index
    module_path_repos = scan.module_path_repos

    resolved: list[tuple] = []
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
        # A framework extractor may describe one declaration twice under one
        # name: the class or method as a ``symbol`` and what it stands for (an
        # ORM model as ``storage``, a relation method as ``association``). The
        # writer then withholds ``target_id`` (two identities), but the two are
        # not rival targets: a reference to the name means the declaration.
        candidates = _without_declaration_views(candidates, fact_index)
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
        properties: dict[str, Any] | None = None,
    ) -> None:
        source_repo, source_kind, source_name = source
        target_repo, target_kind, target_name = target
        source_id = fact_node_id(source_repo, source_kind, source_name)
        target_id = fact_node_id(target_repo, target_kind, target_name)
        edge_key = (str(source_id), str(target_id), relationship_name)
        if edge_key in edge_keys:
            return
        edge_keys.add(edge_key)
        resolved.append(
            (
                source_id,
                target_id,
                relationship_name,
                {
                    **(properties or {}),
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
    for fact in facts:
        kind = fact.get("kind")
        name = fact.get("name")
        if not _is_mappable_fact(kind, name):
            continue
        source_repo = _fact_repo(fact, fallback_repo)

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
            yield from resolved
            resolved.clear()

    skipped += _add_client_route_edges(scan.client_routes, enola_id_index, fact_index, _add_edge)
    yield from resolved
    resolved.clear()

    # Enola's query graph connects a dependency import to the modules which
    # contain each side. Materialize the same bridge so Cognee traversals can
    # move between modules without re-reading the snapshot.
    modules_by_name: dict[str, list[tuple[str, str]]] = {}
    for (repo, module_name), module_name_value in module_index.items():
        modules_by_name.setdefault(module_name, []).append((repo, module_name_value))

    for source_repo, file_path, import_targets in scan.dependencies:
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

        for target_name in import_targets:
            target_module: tuple[str, str] | None = None
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
            target_repo, target_module_name = target_module
            if source_repo == target_repo and source_module == target_module_name:
                continue
            _add_edge(
                (source_repo, "module", source_module),
                (target_repo, "module", target_module_name),
                "imports",
            )
            yield from resolved
            resolved.clear()

    # Enola derives type membership from qualified symbol names. Persisting the
    # edge makes type-level impact analysis include method/function callers.
    owner_symbol_kinds = {"struct", "interface", "class", "type"}
    for source_repo, symbol_name in scan.method_symbols:
        owner_name = symbol_name.rsplit(".", 1)[0]
        owner = fact_index.get((source_repo, "symbol", owner_name))
        if owner is None or owner.symbol_kind not in owner_symbol_kinds:
            continue
        _add_edge(
            (source_repo, "symbol", owner_name),
            (source_repo, "symbol", symbol_name),
            "has_method",
        )
        yield from resolved
        resolved.clear()

    return skipped


@dataclass(frozen=True)
class _SnapshotEnd:
    """What ``extract_code_graph`` yields after its last batch of DataPoints.

    The end of the stream, made explicit: the edge step runs once, on this,
    with what it needs of the whole snapshot. ``source`` reads the facts again
    (``EnolaSnapshot`` streams facts.jsonl; a cluster's ``ParsedSnapshot`` holds
    them), ``fallback_repo`` is the one the mapping used.
    """

    source: Any
    repo_path: str | Path | None
    repo_scope: str | None
    fallback_repo: str
    snapshot_id: str | None


async def extract_code_graph(
    data: Any = None,
    repo_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    timeout: float = 600.0,
    repo_scope: str | None = None,
    node_set: list[str] | None = None,
    snapshot: ParsedSnapshot | None = None,
    batch_size: int = DEFAULT_GRAPH_STREAM_BATCH_SIZE,
) -> AsyncIterator[list[DataPoint] | _SnapshotEnd]:
    """Run enola on repo_path (or reuse an existing snapshot) and stream its DataPoints.

    Yields lists of at most ``batch_size`` DataPoints, then one ``_SnapshotEnd``.
    facts.jsonl is read line by line and never held whole (a cluster's
    ``snapshot`` is already in memory). ``node_set`` tags every node (see
    ``iter_fact_data_points``). An unchanged snapshot yields nothing at all,
    but only for an untagged repo: a tagged one always reloads, so a tag change
    (or a run that crashed mid-retag) is never skipped.

    repo_scope loads one member of a multi-repo snapshot (snapshot_dir is then
    the cluster's snapshot and repo_path the member's directory): the skip
    check reads that member's repository node and only its facts are mapped.
    ``snapshot`` is that snapshot already parsed (a cluster load parses it
    once for all its members); it stands in for snapshot_dir.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    # When used as the first pipeline task, the pipeline payload arrives as the
    # first positional argument; accept a repo path there, ignore anything else.
    if repo_path is None and isinstance(data, (str, Path)):
        repo_path = data

    if snapshot is not None:
        source = snapshot
    else:
        if snapshot_dir is None:
            if repo_path is None:
                raise ValueError("extract_code_graph requires repo_path or snapshot_dir.")
            snapshot_dir = await run_enola_generate(repo_path, timeout=timeout)
        source = EnolaSnapshot(snapshot_dir)
    receipt = source.receipt

    if receipt:
        logger.info(
            "enola snapshot provenance: version=%s format_version=%s snapshot_id=%s facts=%s",
            receipt.get("enola_version"),
            receipt.get("format_version"),
            receipt.get("snapshot_id"),
            receipt.get("fact_count"),
        )

    fallback_repo = _resolve_fallback_repo(source.iter_facts(), repo_path)
    snapshot_id = snapshot_identity(source.snapshot_dir, receipt)
    if snapshot_id is not None:
        skip_repo = repo_scope or fallback_repo
        try:
            stored_id, stored_version, stored_tagged = await _stored_repository_state(skip_repo)
        except Exception as error:
            # The skip check is an optimization; never let it break ingestion.
            logger.warning(
                "Could not read the stored snapshot id (%s); loading fully.", error, exc_info=True
            )
            stored_id, stored_version, stored_tagged = None, None, False
        if (
            stored_id == snapshot_id
            and stored_version == CODE_GRAPH_MAPPING_VERSION
            and not node_set
            and not stored_tagged
        ):
            logger.info(
                "Code graph for '%s' already matches snapshot %s; skipping load.",
                skip_repo,
                snapshot_id,
            )
            return

    batch: list[DataPoint] = []
    mapped = 0
    for point in iter_fact_data_points(
        source.iter_facts(),
        repo_path=repo_path,
        repo_scope=repo_scope,
        node_set=node_set,
        fallback_repo=fallback_repo,
    ):
        batch.append(point)
        mapped += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
    logger.info("Mapped %d enola fact(s) to %d data point(s).", source.fact_count, mapped)

    yield _SnapshotEnd(
        source=source,
        repo_path=repo_path,
        repo_scope=repo_scope,
        fallback_repo=fallback_repo,
        snapshot_id=snapshot_id,
    )


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


async def _stored_repository_state(repo: str) -> tuple[str | None, int | None, bool]:
    """(last_snapshot_id, last_mapping_version, whether it carries a node_set).

    Read off the repository node. A node stamped before the mapping version
    existed has none, so its unchanged snapshot is loaded once more.

    The snapshot id lives on the CodeRepository node in the graph itself, not
    in the relational metastore: this pipeline persists no Data row to key it
    on, and a marker stored with the graph can never outlive it.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    graph_engine = await get_graph_engine()
    node = await graph_engine.get_node(str(fact_node_id(repo, "repository", repo)))
    stored_id = _read_repo_node_property(node, "last_snapshot_id")
    stored_version = _read_repo_node_property(node, "last_mapping_version")
    tags = _read_repo_node_property(node, "belongs_to_set")
    if isinstance(tags, str):
        tags = json.loads(tags)
    return (
        stored_id if isinstance(stored_id, str) and stored_id else None,
        _as_mapping_version(stored_version),
        bool(tags),
    )


def _as_mapping_version(value: Any) -> int | None:
    """The stored mapping version as an int, whatever scalar type the adapter returned."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


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


# The node properties the code graph steps read off the pre-write graph: the
# sweep and detag match a node by type and repo (name for CodeRepository), the
# detag reads tags and NodeSet names. fact_hash is kept in existing_hashes.
_RETAINED_NODE_PROPERTIES = ("type", "name", "repo", "belongs_to_set")


class _CodeGraphState:
    """The pre-write graph, cut down to what the code graph load reads.

    - ``existing_hashes``: ``{node_id: fact_hash}`` of every node, for the
      node delta.
    - ``existing_node_ids``: every node id, for a cluster member's
      ``known_ids`` (an edge's other endpoint may be any node).
    - ``existing_nodes``: ``(node_id, properties)`` of code-graph and NodeSet
      nodes only, with ``_RETAINED_NODE_PROPERTIES`` only, for the sweep and
      detag.
    - ``existing_edge_keys``: ``(source, target, relationship)`` of edges whose
      source is a code-graph node. Every use starts there: the typed-edge diff
      (edges between code nodes), the stale-edge sweep (both endpoints are
      current code nodes) and the ``belongs_to_set`` detag (from a code node).
    """

    def __init__(self) -> None:
        self.existing_hashes: dict[str, Any] = {}
        self.existing_node_ids: set[str] = set()
        self.existing_nodes: list[tuple[str, dict]] = []
        self.existing_edge_keys: set[tuple[str, str, str]] = set()


async def _read_code_graph_state(graph_engine, batch_size: int) -> _CodeGraphState:
    """Read the graph a page at a time, keeping only what the code graph load reads.

    The stream is drained completely before the caller writes anything: the
    node delta, edge diff and sweep compare against the graph as it was, and
    the stream has no snapshot across pages, so a page read after a write
    could see rows this load wrote.
    """
    code_types = _code_graph_node_types()
    retained_types = code_types | {NodeSet.__name__}
    code_node_ids: set[str] = set()
    state = _CodeGraphState()
    async for nodes, edges in graph_engine.stream_graph_data(batch_size=batch_size):
        for node_id, properties in nodes:
            node_id = str(node_id)
            state.existing_node_ids.add(node_id)
            if not isinstance(properties, dict):
                continue
            state.existing_hashes[node_id] = properties.get("fact_hash")
            node_type = properties.get("type")
            if node_type not in retained_types:
                continue
            if node_type in code_types:
                code_node_ids.add(node_id)
            retained = {
                key: properties[key] for key in _RETAINED_NODE_PROPERTIES if key in properties
            }
            state.existing_nodes.append((node_id, retained))
        for source, target, relationship, _properties in edges:
            source = str(source)
            if source in code_node_ids:
                state.existing_edge_keys.add((source, str(target), relationship))
    return state


_DELTA_SAMPLE_LIMIT = 20


def _delta_samples(names: list[str]) -> list[str]:
    return sorted(names)[:_DELTA_SAMPLE_LIMIT]


@dataclass
class _CodeGraphLoad:
    """One load's state across its batches: the graph as read before the first
    write, and the node delta the batches add up to."""

    graph_state: _CodeGraphState
    nodes_added: int = 0
    nodes_updated: int = 0
    nodes_unchanged: int = 0
    samples_added: list[str] = field(default_factory=list)
    samples_updated: list[str] = field(default_factory=list)


async def load_code_graph(
    data: Any = None,
    repo_path: str | Path | None = None,
    ctx: Optional["PipelineContext"] = None,
    snapshot_dir: str | Path | None = None,
    timeout: float = 600.0,
    repo_scope: str | None = None,
    node_set: list[str] | None = None,
    snapshot: ParsedSnapshot | None = None,
    graph_only: bool = True,
    batch_size: int = DEFAULT_GRAPH_STREAM_BATCH_SIZE,
    graph_read_batch_size: int = DEFAULT_GRAPH_STREAM_BATCH_SIZE,
) -> Any:
    """Load one repository's code graph: the one driver of the code graph steps.

    ``extract_code_graph`` streams the snapshot's DataPoints ``batch_size`` at
    a time; each batch is stored by ``add_code_graph_data_points``, and the
    ``_SnapshotEnd`` after the last one runs ``add_code_graph_edges`` once.
    Before the first write the graph is read once (a stream of
    ``graph_read_batch_size`` rows a page), and every step compares against
    that read: nothing this load writes can show up in it. An unchanged
    snapshot yields nothing, so nothing is read or written. A larger
    ``batch_size`` holds more of the snapshot at once and makes fewer writes.

    Returns ``data`` unchanged, so it can stand as a pipeline task.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if graph_read_batch_size < 1:
        raise ValueError("graph_read_batch_size must be >= 1")

    load: _CodeGraphLoad | None = None
    async for item in extract_code_graph(
        data,
        repo_path=repo_path,
        snapshot_dir=snapshot_dir,
        timeout=timeout,
        repo_scope=repo_scope,
        node_set=node_set,
        snapshot=snapshot,
        batch_size=batch_size,
    ):
        if load is None:
            graph_engine = await get_graph_engine()
            load = _CodeGraphLoad(
                graph_state=await _read_code_graph_state(graph_engine, graph_read_batch_size)
            )
        if isinstance(item, _SnapshotEnd):
            await add_code_graph_edges(
                item, load, ctx=ctx, node_set=node_set, batch_size=batch_size
            )
        else:
            await add_code_graph_data_points(item, load, ctx=ctx, graph_only=graph_only)
    return data


async def add_code_graph_data_points(
    data_points: list[DataPoint],
    load: "_CodeGraphLoad",
    ctx: Optional["PipelineContext"] = None,
    graph_only: bool = True,
) -> None:
    """Store one batch of code graph nodes: only the new and the changed ones.

    Delta writes against the graph ``load`` read before the first batch: a
    fact whose fact_hash is new or changed is stored, an unchanged one is not
    touched. Repository nodes are always rewritten (they carry the snapshot
    stamp) and are not counted as content changes. The counts and samples add
    up in ``load`` for the stamp the edge step writes.

    The storage rollback ledger requires a persisted data item id: the full
    context is passed when one is available, otherwise the nodes are stored
    without ledger provenance. graph_only keeps the deterministic default free
    of embedding calls; set it to False to also build vector indexes for
    completion-based search types.
    """
    from cognee.tasks.storage.add_data_points import add_data_points

    existing_hashes = load.graph_state.existing_hashes
    to_write: list[DataPoint] = []
    added: list[str] = []
    updated: list[str] = []
    for point in data_points:
        if isinstance(point, CodeRepository):
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
            load.nodes_unchanged += 1
    load.nodes_added += len(added)
    load.nodes_updated += len(updated)
    load.samples_added = _delta_samples(load.samples_added + added)
    load.samples_updated = _delta_samples(load.samples_updated + updated)

    storage_ctx = ctx if _pipeline_data_id(ctx) is not None else None
    try:
        if to_write:
            await add_data_points(to_write, ctx=storage_ctx, graph_only=graph_only)
    finally:
        # Storage can fail after a partial graph write. Invalidate even on an
        # exception so no pre-write snapshot survives that ambiguous outcome.
        _invalidate_code_graph_snapshot(ctx)


async def add_code_graph_edges(
    end: _SnapshotEnd,
    load: "_CodeGraphLoad",
    ctx: Optional["PipelineContext"] = None,
    node_set: list[str] | None = None,
    batch_size: int = DEFAULT_GRAPH_STREAM_BATCH_SIZE,
) -> None:
    """Insert typed relation edges (calls/imports/...) once every node batch is stored.

    Relation names are dynamic, so they cannot be expressed as DataPoint field
    references; instead they are written directly through the graph engine,
    following the extract_dlt_fk_edges precedent. Afterwards, stale nodes and
    edges from earlier ingestions of the same repos are swept, a node_set
    re-tag (if any) is detagged off the nodes that no longer carry it, and the
    snapshot identity is stamped on the repository node so the next unchanged
    ingestion can skip entirely.

    With repo_scope (one member of a multi-repo snapshot) only edges touching
    that repository are written, and only when the other endpoint is already
    in the graph: an edge into a member loaded later is written by that
    member's load, so the members can load in any order, one at a time.

    The facts are read again for this (``end.source``): once into a compact
    index (``_scan_snapshot``), once more for their relations. The edges are
    resolved and written ``batch_size`` new edges at a time; their keys are
    the one set the resolver deduplicates on and the sweep reads. A cluster's
    edges were resolved once for all members and are only batched.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    repo_scope = end.repo_scope
    graph_state = load.graph_state
    existing_edge_keys = graph_state.existing_edge_keys
    logger.info(
        "Code graph node delta: %d added, %d updated, %d unchanged.",
        load.nodes_added,
        load.nodes_updated,
        load.nodes_unchanged,
    )

    try:
        graph_engine = await get_graph_engine()

        scan = _scan_snapshot(end.source.iter_facts(), end.fallback_repo)
        edge_keys: set[tuple[str, str, str]] = set()
        precomputed = getattr(end.source, "edges", None)
        if precomputed is not None:
            edges = _resolved(precomputed)
        else:
            edges = iter_code_graph_edges(scan, end.source.iter_facts(), edge_keys)

        scope_ids: set[str] = set()
        known_ids: set[str] = set()
        if repo_scope is not None:
            scope_ids = scan.scoped_node_ids(repo_scope)
            known_ids = scope_ids | graph_state.existing_node_ids

        # Register the edges added by this run in the relational rollback
        # ledger (when a pipeline context with a persisted data item is
        # available) so pipeline rollback can clean them up. Custom pipelines
        # may use arbitrary payloads, such as the repository path used by the
        # code graph example.
        data_id = _pipeline_data_id(ctx)
        ledger_ctx = (
            ctx
            if ctx is not None
            and getattr(ctx, "user", None) is not None
            and getattr(ctx, "dataset", None) is not None
            and data_id is not None
            and getattr(ctx, "pipeline_run_id", None) is not None
            else None
        )

        async def write(batch: list[tuple]) -> None:
            await graph_engine.add_edges(batch)
            if ledger_ctx is not None:
                from cognee.modules.graph.methods import upsert_edges

                # Per batch, right after its write: a failure later in the
                # run leaves every batch already written in the ledger.
                await upsert_edges(
                    batch,
                    tenant_id=ledger_ctx.user.tenant_id,
                    user_id=ledger_ctx.user.id,
                    dataset_id=ledger_ctx.dataset.id,
                    data_id=data_id,
                    pipeline_run_id=ledger_ctx.pipeline_run_id,
                )

        # Delta writes: only edges the graph does not already have, written a
        # batch at a time as they resolve.
        batch: list[tuple] = []
        new_count = 0
        while True:
            try:
                edge = next(edges)
            except StopIteration as stop:
                skipped = stop.value
                break
            key = (str(edge[0]), str(edge[1]), edge[2])
            edge_keys.add(key)
            if repo_scope is not None and not (
                (key[0] in scope_ids or key[1] in scope_ids)
                and key[0] in known_ids
                and key[1] in known_ids
            ):
                continue
            if key in existing_edge_keys:
                continue
            batch.append(edge)
            new_count += 1
            if len(batch) == batch_size:
                await write(batch)
                batch = []
        if batch:
            await write(batch)
        logger.info("Resolved %d code graph edge(s), skipped %d.", len(edge_keys), skipped)
        logger.info(
            "Code graph edge delta: %d new, %d already present.",
            new_count,
            len(edge_keys) - new_count,
        )

        # Insert-then-sweep: with the current snapshot fully merged, remove
        # what previous ingestions derived that this snapshot no longer does.
        # Runs even with zero edges — a shrunken repo still needs its sweep.
        nodes_removed, edges_removed, samples_removed = await _sweep_stale_code_graph(
            graph_engine, scan, edge_keys, graph_state.existing_nodes, existing_edge_keys
        )

        # Retag: strip node_set names this repo carried before but not anymore.
        # A no-op (zero graph calls) when node_set is unchanged, including the
        # untagged case, so the per-file CODE route and an untagged repository
        # (node_set=None) never pay for this.
        detagged = await _detag_stale_code_graph(
            graph_engine,
            scan,
            graph_state.existing_nodes,
            existing_edge_keys,
            node_set,
            repo_scope=repo_scope,
        )

        delta = {
            "nodes_added": load.nodes_added,
            "nodes_updated": load.nodes_updated,
            "nodes_unchanged": load.nodes_unchanged,
            "samples_added": load.samples_added,
            "samples_updated": load.samples_updated,
            "edges_added": new_count,
            "edges_removed": edges_removed,
            "nodes_removed": nodes_removed,
            "samples_removed": samples_removed,
            "node_set_detagged": detagged,
            "snapshot_id": end.snapshot_id,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }

        # Stamp last: only a load that added, swept, detagged, and got here may
        # record its snapshot id, so a crashed run can never be skipped-past later.
        await _stamp_snapshot_identity(
            graph_engine,
            scan,
            end.repo_path,
            end.snapshot_id,
            delta,
            receipt=end.source.receipt,
            repo_scope=repo_scope,
            node_set=node_set,
        )
    finally:
        # Direct edge writes, sweeps, and ledger writes may partially succeed.
        _invalidate_code_graph_snapshot(ctx)


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
    scan: _SnapshotScan,
    current_edge_keys: set[tuple[str, str, str]],
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

    snapshot_repos = scan.repos
    current_ids = scan.node_repos

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

    # Structural containment edges are written by add_data_points from the
    # DataPoint part_of field: each node's to its repository node.
    stale_edges = [
        EdgeIdentity(source_id=source, target_id=target, relationship_name=relationship)
        for source, target, relationship in existing_edge_keys
        if source in current_ids
        and target in current_ids
        and (source, target, relationship) not in current_edge_keys
        and not (relationship == "part_of" and current_ids[source] == target)
    ]
    if stale_edges:
        for start in range(0, len(stale_edges), _SWEEP_CHUNK_SIZE):
            await graph_engine.delete_edge_triples(stale_edges[start : start + _SWEEP_CHUNK_SIZE])
        logger.info("Swept %d stale code graph edge(s).", len(stale_edges))

    return len(stale_node_ids), len(stale_edges), _delta_samples(stale_node_names)


async def _detag_stale_code_graph(
    graph_engine,
    scan: _SnapshotScan,
    existing_nodes: list[tuple],
    existing_edge_keys: set,
    node_set: list[str] | None,
    repo_scope: str | None = None,
) -> list[str]:
    """Strip node_set tags this repo's surviving code nodes carry but should not.

    Stale tags are read off the nodes themselves (their ``belongs_to_set``
    edges and property), so a tag left behind by a crashed run is found too.
    A node rewrite never removes edges, so stale edges are deleted on every
    backend; the property is only stale on Neo4j, whose rewrite unions it.

    Edges are stale by NodeSet id, the property by exact name: a respelling
    ("Team-A" -> "team-a") keeps the shared edge but drops the old spelling.

    With repo_scope (one member of a multi-repo snapshot) only that member's
    nodes are considered: the other members carry their own rows' tags, which
    are not stale just because they differ from this row's.
    """
    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity

    if repo_scope is not None:
        snapshot_repos = {repo_scope}
        current_ids = scan.scoped_node_ids(repo_scope)
    else:
        snapshot_repos = scan.repos
        current_ids = scan.node_repos
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

    nodeset_names = {
        str(node_id): properties.get("name")
        for node_id, properties in existing_nodes
        if isinstance(properties, dict) and properties.get("type") == "NodeSet"
    }
    property_names: set = set()
    for node_id, properties in existing_nodes:
        if str(node_id) in code_node_id_set and isinstance(properties, dict):
            tags = properties.get("belongs_to_set")
            if isinstance(tags, list):
                property_names.update(tag for tag in tags if isinstance(tag, str))
    edge_names = {
        nodeset_names[target]
        for source, target, relationship in existing_edge_keys or ()
        if relationship == "belongs_to_set"
        and source in code_node_id_set
        and isinstance(nodeset_names.get(target), str)
    }

    current = set(node_set or [])
    current_ids_by_name = {NodeSet.id_for(name) for name in current}
    stale_edge_names = sorted(
        name
        for name in edge_names | property_names
        if NodeSet.id_for(name) not in current_ids_by_name
    )
    stale_property_names = sorted(property_names - current)
    if not stale_edge_names and not stale_property_names:
        return []

    stale_edges = [
        EdgeIdentity(
            source_id=node_id,
            target_id=str(NodeSet.id_for(name)),
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
    scan: _SnapshotScan,
    repo_path: str | Path | None,
    snapshot_id: str | None,
    delta: dict | None = None,
    receipt: dict | None = None,
    repo_scope: str | None = None,
    node_set: list[str] | None = None,
) -> None:
    """Record the loaded snapshot's identity, delta, receipt and tag on the repository nodes.

    With repo_scope only that member is stamped: the other members of the
    snapshot have not loaded yet, and a stamp would make their load skip.

    This rewrites the CodeRepository node through ``add_nodes``, so it carries
    the tag itself, as names (the stored shape): omitting it would erase the
    tag on adapters that overwrite node properties wholesale.
    """
    if snapshot_id is None:
        return
    fallback_repo = repo_scope or scan.fallback_repo
    repos = {repo_scope} if repo_scope else scan.repos
    last_receipt = receipt_projection(receipt)
    source_node_set = ", ".join(node_set) if node_set else None
    repositories = [
        CodeRepository(
            id=fact_node_id(repo, "repository", repo),
            name=repo,
            path=str(repo_path) if repo_path and repo == fallback_repo else repo,
            last_snapshot_id=snapshot_id,
            last_mapping_version=CODE_GRAPH_MAPPING_VERSION,
            last_delta=delta,
            last_receipt=last_receipt,
            belongs_to_set=list(node_set) if node_set else None,
            source_node_set=source_node_set,
        )
        for repo in sorted(repos)
    ]
    await graph_engine.add_nodes(repositories)


def get_code_graph_tasks(
    repo_path: str | Path,
    snapshot_dir: str | Path | None = None,
    timeout: float = 600.0,
    index_vectors: bool = False,
    repo_scope: str | None = None,
    snapshot: ParsedSnapshot | None = None,
) -> list[Task]:
    """The enola code graph pipeline as a task list: one task, ``load_code_graph``.

    index_vectors is opt-in because SearchType.CODE uses graph indexes only.
    Enable it when the same code facts must also feed semantic/completion
    retrievers, which may require an embedding provider API key. repo_scope
    loads one member of the multi-repo snapshot in snapshot_dir; ``snapshot``
    is that snapshot already parsed, shared by the members' task lists.
    """
    return [
        Task(
            load_code_graph,
            repo_path=repo_path,
            snapshot_dir=snapshot_dir,
            timeout=timeout,
            repo_scope=repo_scope,
            snapshot=snapshot,
            graph_only=not index_vectors,
        ),
    ]
