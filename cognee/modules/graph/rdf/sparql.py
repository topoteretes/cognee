"""Read-only SPARQL over one dataset's memory graph.

The RDF view is not a live store: ``export_memory_graph_to_rdf`` reads the whole
dataset graph (``get_graph_data``) and loads it into an in-memory rdflib store,
then the query runs there. That keeps the feature engine-agnostic (Kuzu, Neo4j,
... all share one SPARQL surface) at the cost of materializing the graph per
request. Three guards keep that cost bounded:

* the query is refused up front when it carries SPARQL Update keywords, so the
  view can never be mutated (and the error names the problem rather than
  surfacing an rdflib parse error), or when it reads anything beyond the view:
  ``SERVICE`` would make the server send an HTTP request to any endpoint, and
  ``FROM`` / ``FROM NAMED`` name graphs to load from a URL or file;
* the dataset's node count is checked *before* the export, and graphs above
  ``SPARQL_MAX_GRAPH_NODES`` are refused with a pointer at the RDF export, which
  streams the same triples without holding a query engine in memory;
* parsing, export and evaluation run in worker threads under one wall-clock
  timeout (``SPARQL_QUERY_TIMEOUT_SECONDS``), so a pathological query cannot pin
  the event loop. Python threads cannot be cancelled, so a timed-out query keeps
  running in the background until it finishes; the caller only stops waiting.

The pure pieces (keyword guard, term conversion, result formatting) take plain
rdflib objects so they are testable without a graph engine.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from uuid import UUID

from pyparsing import ParseException, ParseResults
from rdflib import BNode, Graph
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.sparql import Query
from rdflib.query import Result

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.graph.rdf.export import DEFAULT_BASE_IRI, graph_data_to_rdf
from cognee.shared.logging_utils import get_logger

logger = get_logger("SPARQL")

DEFAULT_LIMIT = 1000
MAX_LIMIT = 10_000
DEFAULT_MAX_GRAPH_NODES = 200_000
DEFAULT_QUERY_TIMEOUT_SECONDS = 30.0

# SPARQL 1.1 Update operations (https://www.w3.org/TR/sparql11-update/). Any of
# these outside a string literal, IRI or comment marks the request as an update.
# ``DELETE`` also covers ``DELETE DATA`` / ``DELETE WHERE``; ``INSERT`` covers
# ``INSERT DATA``. ``WITH`` and ``USING`` only exist in update grammar but are
# harmless without a verb, so they are not listed.
_UPDATE_KEYWORDS = ("INSERT", "DELETE", "LOAD", "CLEAR", "DROP", "CREATE", "ADD", "MOVE", "COPY")

# Parse-tree nodes that reach beyond the dataset view: SERVICE sends an HTTP request
# to the endpoint it names, FROM / FROM NAMED name graphs to load from a URL or file.
_EXTERNAL_ACCESS_NODES = {
    "ServiceGraphPattern": "SERVICE",
    "DatasetClause": "FROM / FROM NAMED",
}

# A keyword counts only as a bare token: not the tail of a variable (``?delete``),
# a prefixed name (``ex:create``, ``create:x``) or a longer word (``created``).
_UPDATE_KEYWORD_RE = re.compile(
    r"(?<![?$:\w-])(" + "|".join(_UPDATE_KEYWORDS) + r")(?![\w:-])",
    re.IGNORECASE,
)

# Lexical regions of a SPARQL query whose bytes carry no keywords: long and
# short string literals, IRIs and comments. Order matters: a ``#`` inside a
# string or IRI (``<http://x/y#z>``) is not a comment, so those alternatives
# come first and the regex consumes them whole.
_SKIP_REGION_RE = re.compile(
    r'"""(?:\\.|[^\\])*?"""'  # long double-quoted string
    r"|'''(?:\\.|[^\\])*?'''"  # long single-quoted string
    r'|"(?:\\.|[^"\\\n])*"'  # short double-quoted string
    r"|'(?:\\.|[^'\\\n])*'"  # short single-quoted string
    r"|<[^<>\"{}|^`\\\s]*>"  # IRI reference
    r"|#[^\n]*",  # comment to end of line
    re.DOTALL,
)


class SparqlQueryError(ValueError):
    """Base for SPARQL requests refused before or while running."""


class SparqlSyntaxError(SparqlQueryError):
    """rdflib could not parse the query."""


class SparqlUpdateNotAllowedError(SparqlQueryError):
    """The query carries SPARQL Update keywords; the RDF view is read-only."""


class SparqlExternalAccessNotAllowedError(SparqlQueryError):
    """The query reads beyond the dataset view (``SERVICE``, ``FROM``, ``FROM NAMED``)."""


class SparqlGraphTooLargeError(SparqlQueryError):
    """The dataset graph exceeds ``SPARQL_MAX_GRAPH_NODES``; use the RDF export."""

    def __init__(self, num_nodes: int, max_nodes: int):
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes
        super().__init__(
            f"Dataset graph has {num_nodes} nodes, above the SPARQL limit of {max_nodes} "
            "(SPARQL_MAX_GRAPH_NODES). The query engine materializes the whole graph in "
            "memory per request; export the graph as RDF instead "
            "(cognee.modules.graph.rdf.serialize_memory_graph) and query it in a triple store."
        )


class SparqlQueryTimeoutError(SparqlQueryError):
    """Export + query did not finish within ``SPARQL_QUERY_TIMEOUT_SECONDS``."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"SPARQL query did not finish within {timeout_seconds:g}s "
            "(SPARQL_QUERY_TIMEOUT_SECONDS). Narrow the query or raise the timeout."
        )


def max_graph_nodes() -> int:
    """Node ceiling for in-memory SPARQL; ``SPARQL_MAX_GRAPH_NODES`` env, default 200k."""
    raw = os.getenv("SPARQL_MAX_GRAPH_NODES")
    return int(raw) if raw else DEFAULT_MAX_GRAPH_NODES


def query_timeout_seconds() -> float:
    """Wall-clock budget per query; ``SPARQL_QUERY_TIMEOUT_SECONDS`` env, default 30."""
    raw = os.getenv("SPARQL_QUERY_TIMEOUT_SECONDS")
    return float(raw) if raw else DEFAULT_QUERY_TIMEOUT_SECONDS


def strip_sparql_comments(query: str) -> str:
    """Drop ``# ...`` comments while leaving string literals and IRIs intact.

    ``<http://example.org/ns#term>`` and ``"issue #42"`` keep their ``#``; only a
    ``#`` outside those regions starts a comment.
    """

    def _keep_or_drop(match: re.Match[str]) -> str:
        return "" if match.group(0).startswith("#") else match.group(0)

    return _SKIP_REGION_RE.sub(_keep_or_drop, query)


def _keyword_scan_text(query: str) -> str:
    """The query with comments, string literals and IRIs blanked out.

    What remains is the structural text where an Update keyword would actually
    be an operation, so ``FILTER(CONTAINS(?s, "delete"))`` is not mistaken for
    ``DELETE``.
    """
    return _SKIP_REGION_RE.sub(" ", query)


def is_update_query(query: str) -> bool:
    """True when the query holds a SPARQL Update keyword outside strings/IRIs/comments."""
    return _UPDATE_KEYWORD_RE.search(_keyword_scan_text(query)) is not None


def assert_read_only(query: str) -> None:
    """Raise ``SparqlUpdateNotAllowedError`` for a query that would mutate the view."""
    match = _UPDATE_KEYWORD_RE.search(_keyword_scan_text(query))
    if match is not None:
        raise SparqlUpdateNotAllowedError(
            f"SPARQL Update is not supported: found '{match.group(1).upper()}'. "
            "Only read-only query forms (SELECT, ASK, CONSTRUCT, DESCRIBE) run against "
            "the dataset graph; write through cognee.add/cognify instead."
        )


def _parse_tree_node_names(node: Any, names: set[str]) -> set[str]:
    """Collect the name of every node in an rdflib SPARQL parse tree."""
    if isinstance(node, CompValue):
        names.add(node.name)
        for value in node.values():
            _parse_tree_node_names(value, names)
    elif isinstance(node, (list, tuple, ParseResults)):
        for item in node:
            _parse_tree_node_names(item, names)
    return names


def _too_complex_error(error: RecursionError) -> SparqlSyntaxError:
    # rdflib's parser and algebra recurse per triple pattern; around 90 patterns
    # exhaust Python's default recursion limit.
    return SparqlSyntaxError(
        "SPARQL query is too long or too deeply nested to process; split it into "
        f"smaller queries ({error})."
    )


def _parse(query: str) -> ParseResults:
    try:
        return parseQuery(query)
    except ParseException as error:
        raise SparqlSyntaxError(f"Invalid SPARQL query: {error}") from error
    except RecursionError as error:
        raise _too_complex_error(error) from error


def assert_no_external_access(parse_tree: ParseResults) -> None:
    """Refuse a parsed query that reads beyond the dataset view.

    Checks the parse tree, not the text, so ``SERVICE`` is found wherever it is
    nested (subquery, ``OPTIONAL``, ``UNION``) and a literal or variable that merely
    spells "service" passes.
    """
    node_names = _parse_tree_node_names(parse_tree, set())
    for node_name, clause in _EXTERNAL_ACCESS_NODES.items():
        if node_name in node_names:
            raise SparqlExternalAccessNotAllowedError(
                f"SPARQL {clause} is not supported: the query runs over this dataset's "
                "graph only and cannot read other graphs or endpoints."
            )


def prepare_query(query: str) -> ParseResults:
    """Check ``query`` and return its parse tree: read-only, parses, no external access.

    Pass the tree to ``compile_query`` against the graph it runs on, so the text is
    parsed once. Raises ``SparqlUpdateNotAllowedError``, ``SparqlSyntaxError`` or
    ``SparqlExternalAccessNotAllowedError``.
    """
    assert_read_only(query)
    parse_tree = _parse(query)
    assert_no_external_access(parse_tree)
    return parse_tree


def compile_query(parse_tree: ParseResults, graph: Graph) -> Query:
    """Compile a ``prepare_query`` tree with ``graph``'s prefixes, as ``Graph.query`` does.

    The RDF view binds ``cognee:`` and ``cprop:`` (``graph_data_to_rdf``), so queries
    may use them without a ``PREFIX`` line.
    """
    try:
        return translateQuery(parse_tree, initNs=dict(graph.namespaces()))
    except RecursionError as error:
        raise _too_complex_error(error) from error
    except Exception as error:
        # rdflib raises a bare Exception for an undeclared prefix.
        if str(error).startswith("Unknown namespace prefix"):
            raise SparqlSyntaxError(f"Invalid SPARQL query: {error}") from error
        raise


def term_to_json(term: Any) -> str | None:
    """Flatten an rdflib term to a plain string for the JSON response.

    IRIs are their string; blank nodes are ``_:<id>``; literals are their
    lexical form (``datatype``/``lang`` are dropped; CONSTRUCT/DESCRIBE results
    keep them because they are serialized as Turtle). An unbound variable in a
    SELECT row is ``None``.
    """
    if term is None:
        return None
    if isinstance(term, BNode):
        return f"_:{term}"
    return str(term)


def format_sparql_result(result: Result, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Turn an rdflib ``Result`` into the JSON body shape for its query form.

    * SELECT: ``{"variables", "rows", "row_count", "truncated"}``, ``rows`` capped
      at ``limit``; ``truncated`` is True when at least one more row existed.
    * ASK: ``{"boolean": bool}``.
    * CONSTRUCT / DESCRIBE: ``{"turtle": str}`` -- the result graph as Turtle.

    rdflib evaluates SELECT bindings lazily, so this walks only ``limit + 1`` rows
    and the caller must run it in the same thread as ``Graph.query``.
    """
    if result.type == "ASK":
        return {"boolean": bool(result.askAnswer)}

    if result.type in ("CONSTRUCT", "DESCRIBE"):
        graph = result.graph
        return {"turtle": graph.serialize(format="turtle") if graph is not None else ""}

    if result.type != "SELECT":
        raise SparqlQueryError(f"Unsupported SPARQL result type: {result.type}")

    variables = [str(variable) for variable in (result.vars or [])]
    rows: list[list[str | None]] = []
    truncated = False
    for row in result:
        if len(rows) >= limit:
            truncated = True
            break
        rows.append([term_to_json(term) for term in row])

    return {
        "variables": variables,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def run_sparql_on_graph_data(
    nodes: list,
    edges: list,
    parse_tree: ParseResults,
    base_iri: str = DEFAULT_BASE_IRI,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Build the RDF view from ``(nodes, edges)`` and run the query on it, synchronously.

    ``parse_tree`` comes from ``prepare_query``. Everything CPU-bound lives here
    (RDF build, compile, evaluate, format) so the async caller can push the whole
    unit onto a worker thread.
    """
    graph = graph_data_to_rdf(nodes, edges, base_iri=base_iri)
    query = compile_query(parse_tree, graph)
    try:
        result = graph.query(query)
        return format_sparql_result(result, limit=limit)
    except RecursionError as error:
        raise _too_complex_error(error) from error


async def query_dataset_sparql(
    dataset_id: UUID,
    owner_id: UUID | None,
    query: str,
    *,
    base_iri: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Run a read-only SPARQL query over one dataset's graph.

    The caller must already have checked that the user may read ``dataset_id``
    (``get_authorized_existing_datasets``/``get_authorized_dataset``); this
    function only scopes the graph engine to that dataset the way search does,
    so under access control each tenant's graph stays isolated.

    Raises:
        ValueError: ``limit`` is outside ``1..MAX_LIMIT``.
        SparqlUpdateNotAllowedError: the query carries Update keywords.
        SparqlSyntaxError: rdflib could not parse the query.
        SparqlExternalAccessNotAllowedError: the query uses ``SERVICE``, ``FROM`` or
            ``FROM NAMED``.
        SparqlGraphTooLargeError: the graph has more nodes than
            ``SPARQL_MAX_GRAPH_NODES``.
        SparqlQueryTimeoutError: parse, export and query together exceeded
            ``SPARQL_QUERY_TIMEOUT_SECONDS``.
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}")

    base_iri = base_iri or DEFAULT_BASE_IRI
    max_nodes = max_graph_nodes()
    timeout = query_timeout_seconds()

    async def _parse_export_and_query() -> dict[str, Any]:
        # Parsing is CPU-bound; it runs before the graph is read, so a refused query
        # costs nothing.
        parse_tree = await asyncio.to_thread(prepare_query, query)

        async with set_database_global_context_variables(dataset_id, owner_id):
            graph_engine = await get_graph_engine()

            num_nodes, _ = await graph_engine.get_graph_counts()
            if num_nodes > max_nodes:
                raise SparqlGraphTooLargeError(num_nodes, max_nodes)

            nodes, edges = await graph_engine.get_graph_data()
            logger.info(
                "SPARQL over dataset %s: %d nodes, %d edges", dataset_id, len(nodes), len(edges)
            )
            return await asyncio.to_thread(
                run_sparql_on_graph_data, nodes, edges, parse_tree, base_iri, limit
            )

    try:
        return await asyncio.wait_for(_parse_export_and_query(), timeout=timeout)
    except asyncio.TimeoutError as error:
        raise SparqlQueryTimeoutError(timeout) from error
