"""Join the memories about a repository's code to the symbols that code lives in now.

Every ``ObservedDeclaration`` of a repository is resolved against the
repository's ``CodeSymbol`` nodes and joined to its symbol by a
``resolves_to`` edge:

* ``exact`` -- a current declaration carries the reference's enola id (a
  symbol with the same name in the same file; a re-keyed id over the same
  name and file counts as exact as well);
* ``moved`` -- no declaration carries the id, and exactly one declaration of
  the repository has the reference's body hash: the code was moved or renamed
  and is followed, because nothing else has that body;
* ``ambiguous`` -- no declaration carries the id and several symbols have a
  declaration with that body: one edge per candidate, each marked ambiguous,
  so the reader sees every candidate and picks none;
* no edge -- nothing in the repository carries the id or the body.

The join is recomputed as a whole, as an edge diff, by ``persist_code_findings``
when memories are stored (the code may already be in the graph) and by
``add_code_graph_edges`` when a repository is ingested (the memories may already
be in the graph; moved code is followed here). Nothing on an
``ObservedDeclaration`` is ever rewritten: the match is on the edge, and
whether the code the author saw is still the code is read at query time
(``observed_declaration_status``). Only the graph engine's filtered read, edge
write and edge delete are used, so every graph backend serves this the same.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_memory.models import RESOLVES_TO_RELATIONSHIP, ObservedDeclaration

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_memory")

_SYMBOL_TYPE = "CodeSymbol"
_OBSERVED_TYPE = ObservedDeclaration.__name__


def _plain(value: Any) -> Any:
    """Backend containers (UUIDs, models, JSON-in-string values) as plain JSON values."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return _plain(json.loads(value))
        except ValueError:
            return value
    return value


def node_properties(raw: Any) -> dict[str, Any]:
    """A stored node's or edge's properties as one flat dict, whichever adapter answered.

    Adapters return the properties flattened, or nested under a ``properties``
    key that may itself be JSON text (the reading the code docs tasks share).
    """
    properties = _plain(raw)
    if not isinstance(properties, dict):
        return {}
    nested = properties.get("properties")
    if isinstance(nested, dict):
        properties = {
            **nested,
            **{key: value for key, value in properties.items() if key != "properties"},
        }
    return properties


@dataclass(frozen=True)
class CurrentDeclaration:
    """One declaration a repository's ``CodeSymbol`` node holds now."""

    symbol_id: str
    name: str
    file: str | None
    enola_id: str | None
    body_hash: str | None


class CodeSymbolIndex:
    """A repository's current declarations by enola id and by body hash.

    Built from stored ``CodeSymbol`` nodes: the versioned ``declarations`` a
    symbol carries (enola >= 0.4.37, one per declaration sharing the name), or
    the node's own ``enola_id`` when the snapshot carried no body hashes.
    """

    def __init__(self) -> None:
        self.by_id: dict[str, CurrentDeclaration] = {}
        self.by_hash: dict[str, list[CurrentDeclaration]] = {}
        self.hashes_by_symbol: dict[str, set[str]] = {}

    @classmethod
    def from_nodes(
        cls, nodes: Iterable[tuple[Any, dict[str, Any]]], repos: set[str]
    ) -> "CodeSymbolIndex":
        """Index the ``CodeSymbol`` nodes of these repositories (other nodes are skipped)."""
        index = cls()
        for node_id, properties in nodes:
            if properties.get("type") != _SYMBOL_TYPE or properties.get("repo") not in repos:
                continue
            for declaration in _current_declarations(str(node_id), properties):
                index.add(declaration)
        return index

    def add(self, declaration: CurrentDeclaration) -> None:
        if declaration.enola_id:
            self.by_id[declaration.enola_id] = declaration
        if declaration.body_hash:
            self.by_hash.setdefault(declaration.body_hash, []).append(declaration)
            self.hashes_by_symbol.setdefault(declaration.symbol_id, set()).add(
                declaration.body_hash
            )


def _current_declarations(symbol_id: str, properties: dict[str, Any]) -> list[CurrentDeclaration]:
    declarations = properties.get("declarations")
    if isinstance(declarations, str):
        # Neo4j stores a list as ``<key>_json`` text and the filtered read
        # returns it as stored; Ladybug returns the list.
        declarations = _loads(declarations)
    if declarations is None:
        declarations = _loads(properties.get("declarations_json"))
    if isinstance(declarations, list) and declarations:
        return [
            CurrentDeclaration(
                symbol_id=symbol_id,
                name=str(properties.get("name") or ""),
                file=declaration.get("file"),
                enola_id=declaration.get("id"),
                body_hash=declaration.get("body_hash"),
            )
            for declaration in declarations
            if isinstance(declaration, dict)
        ]
    return [
        CurrentDeclaration(
            symbol_id=symbol_id,
            name=str(properties.get("name") or ""),
            file=properties.get("file_path"),
            enola_id=properties.get("enola_id"),
            body_hash=None,
        )
    ]


def _loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


def resolve_observed_declaration(
    observed: dict[str, Any], index: CodeSymbolIndex
) -> list[tuple[str, str]]:
    """``[(symbol node id, match)]`` for one observed declaration; empty when nothing matches."""
    exact = index.by_id.get(str(observed.get("enola_id") or ""))
    if exact is not None:
        return [(exact.symbol_id, "exact")]
    candidates = index.by_hash.get(str(observed.get("body_hash") or ""), [])
    symbols = sorted({candidate.symbol_id for candidate in candidates})
    if len(symbols) == 1:
        candidate = candidates[0]
        same_place = candidate.name == observed.get("name") and candidate.file == observed.get(
            "file"
        )
        return [(symbols[0], "exact" if same_place else "moved")]
    return [(symbol_id, "ambiguous") for symbol_id in symbols]


def observed_declaration_status(
    observed: dict[str, Any], matches: list[tuple[str, str]], index: CodeSymbolIndex
) -> str:
    """Whether the code a finding's author saw is still the code: ``current``, ``changed``, ``ambiguous`` or ``gone``.

    ``current`` when the resolved symbol holds a declaration with the observed
    body hash (moved code included), ``changed`` when it holds the symbol but
    not that body, ``gone`` when no symbol of the repository carries the
    declaration's id or body.
    """
    if not matches:
        return "gone"
    if any(match == "ambiguous" for _symbol_id, match in matches):
        return "ambiguous"
    symbol_id = matches[0][0]
    hashes = index.hashes_by_symbol.get(symbol_id, set())
    return "current" if observed.get("body_hash") in hashes else "changed"


async def link_observed_declarations(
    graph_engine,
    repos: set[str],
    ctx: Optional["PipelineContext"] = None,
    *,
    existing_nodes: Iterable[tuple[Any, Any]] | None = None,
) -> dict[str, int]:
    """Make the ``resolves_to`` edges of these repositories' observed declarations current.

    One read of the repositories' ``ObservedDeclaration`` and ``CodeSymbol``
    nodes with the edges between them; edges that no longer hold (the symbol
    changed, the match changed) are deleted and the missing ones written, so
    an unchanged graph costs no write. Returns the counts per match plus
    ``unmatched``.

    A repository no memory names costs no read: ``existing_nodes``, when the
    caller holds a read of the graph (the code graph load reads it before
    writing), says so without one; otherwise the declarations are read alone
    first.
    """
    repos = {repo for repo in repos if repo}
    if not repos:
        return {}
    if existing_nodes is None:
        existing_nodes, _edges = await graph_engine.get_filtered_graph_data(
            [{"type": [_OBSERVED_TYPE]}], include_edges=False
        )
    if not any(_names_repo(properties, repos) for _node_id, properties in existing_nodes):
        return {}

    raw_nodes, raw_edges = await graph_engine.get_filtered_graph_data(
        [{"type": [_OBSERVED_TYPE, _SYMBOL_TYPE]}]
    )
    nodes = [(str(node_id), node_properties(properties)) for node_id, properties in raw_nodes]
    index = CodeSymbolIndex.from_nodes(nodes, repos)
    observed = {
        node_id: properties
        for node_id, properties in nodes
        if properties.get("type") == _OBSERVED_TYPE and properties.get("repo") in repos
    }

    existing: dict[tuple[str, str], str | None] = {}
    for item in raw_edges:
        if len(item) < 3 or str(item[2]) != RESOLVES_TO_RELATIONSHIP:
            continue
        source, target = str(item[0]), str(item[1])
        if source in observed:
            properties = node_properties(item[3]) if len(item) > 3 else {}
            existing[(source, target)] = properties.get("match")

    desired: dict[tuple[str, str], str] = {}
    counts = {"exact": 0, "moved": 0, "ambiguous": 0, "unmatched": 0}
    for node_id, properties in observed.items():
        matches = resolve_observed_declaration(properties, index)
        if not matches:
            counts["unmatched"] += 1
        for symbol_id, match in matches:
            desired[(node_id, symbol_id)] = match
            counts[match] += 1

    stale = [key for key, match in existing.items() if desired.get(key) != match]
    new_edges = [
        _resolves_to_edge(source, target, match)
        for (source, target), match in desired.items()
        if existing.get((source, target)) != match
    ]
    if stale:
        from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity

        await graph_engine.delete_edge_triples(
            [
                EdgeIdentity(
                    source_id=source, target_id=target, relationship_name=RESOLVES_TO_RELATIONSHIP
                )
                for source, target in stale
            ]
        )
    if new_edges:
        await graph_engine.add_edges(new_edges)
        await _register_edges(new_edges, ctx)
    logger.info(
        "Code memory join for %s: %d exact, %d moved, %d ambiguous, %d unmatched "
        "(%d edge(s) written, %d removed).",
        ", ".join(sorted(repos)),
        counts["exact"],
        counts["moved"],
        counts["ambiguous"],
        counts["unmatched"],
        len(new_edges),
        len(stale),
    )
    return counts


def _names_repo(properties: Any, repos: set[str]) -> bool:
    """Whether a stored node is an observed declaration of one of these repositories."""
    properties = node_properties(properties)
    return properties.get("type") == _OBSERVED_TYPE and properties.get("repo") in repos


def _resolves_to_edge(source: str, target: str, match: str) -> tuple[str, str, str, dict[str, Any]]:
    """One edge tuple in the shape add_code_graph_edges writes."""
    return (
        source,
        target,
        RESOLVES_TO_RELATIONSHIP,
        {
            "match": match,
            "source_node_id": source,
            "target_node_id": target,
            "relationship_name": RESOLVES_TO_RELATIONSHIP,
            "edge_text": RESOLVES_TO_RELATIONSHIP.replace("_", " "),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


async def _register_edges(edges: list[tuple], ctx: Optional["PipelineContext"]) -> None:
    """Record the edges in the relational rollback ledger, as add_code_graph_edges does.

    Only a run over a persisted data item has a ledger row to key them on: the
    ingest of a code repository, or a session window's persist. A memify run
    over the ``[{}]`` seed has none and relies on graph provenance.
    """
    from cognee.infrastructure.databases.provenance import data_item_id

    data_id = data_item_id(getattr(ctx, "data_item", None))
    if (
        ctx is None
        or getattr(ctx, "user", None) is None
        or getattr(ctx, "dataset", None) is None
        or data_id is None
        or getattr(ctx, "pipeline_run_id", None) is None
    ):
        return
    from cognee.modules.graph.methods import upsert_edges

    await upsert_edges(
        edges,
        tenant_id=ctx.user.tenant_id,
        user_id=ctx.user.id,
        dataset_id=ctx.dataset.id,
        data_id=data_id,
        pipeline_run_id=ctx.pipeline_run_id,
    )
