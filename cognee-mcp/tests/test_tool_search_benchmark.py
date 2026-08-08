"""Benchmark FastMCP's tool-search transforms against a hand-rolled alternative.

Exists to answer "should we have written this ourselves?" with numbers rather
than taste. Three things get measured over synthetic catalogs of mock tools:

* **Context cost** — bytes/tokens of the ``tools/list`` payload an agent pays on
  every turn. This is the entire reason for gating the surface.
* **Latency** — index build plus per-query time, i.e. what gating costs at call
  time.
* **Retrieval quality** — recall@k and MRR over labeled paraphrase queries,
  comparing FastMCP's BM25 and regex transforms against ``NaiveSubstringSearch``
  below, which is roughly the "v1 lexical scoring" a custom implementation would
  have started from.

Run with output:  pytest tests/test_tool_search_benchmark.py -s
Assertions here are deliberately loose (order-of-magnitude, not thresholds) so
the suite doesn't turn into a flaky performance gate. The printed tables are the
deliverable.

Measured conclusion (fastmcp 3.4.5, 500 mock tools, 10 paraphrase queries):

* Gating cuts the ``tools/list`` payload by 67% at today's 11 tools, 92% at 50,
  and 99% at 500 (1199 → 393 tokens, 47329 → 393). The gated payload is
  *constant* in catalog size, which is the actual win.
* **max_results dominates the choice of ranker.** At k=5 the hand-rolled matcher
  looks better on recall (70% vs 60%); at k=10 BM25 pulls ahead (80% vs 70%) and
  both plateau by k=15. BM25's higher MRR (0.55 vs 0.51) was the tell: it was
  placing the right tool just outside a 5-wide window. Any recall comparison at
  a single k says more about the cutoff than about the ranker.
* Recall, not ranking, is the metric to optimise: the agent receives every
  returned schema and selects for itself, so a tool absent from the window is
  unrecoverable while its rank inside the window costs almost nothing. Hence
  ``server.TOOL_SEARCH_MAX_RESULTS = 10``, sitting at the knee of the sweep.
* Both rankers are purely lexical and plateau near 80%. The residual misses are
  vocabulary failures no ranker fixes — "wipe out" vs "permanently remove", and
  plural/singular mismatches, since BM25's tokenizer does **no stemming** and it
  drops zero-scoring tools outright ("tenants" never reaches ``list_tenant``).
  So tool descriptions are the lever for recall, not k and not the ranker; a
  decisive jump beyond that needs embeddings.
* BM25 costs ~3 ms/query at 500 tools versus ~0.2 ms naive. Ten times slower in
  relative terms, irrelevant next to an LLM round trip.

So: use FastMCP's transform. It wins on the metric that matters once k is tuned,
and costs zero lines of our own search code. If we later want semantic matching,
``BaseSearchTransform`` leaves ``_search()`` abstract — we can swap the ranking
function for cognee's embeddings and keep the synthetic tools, the call_tool
proxy, and the visibility/auth composition.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP
from fastmcp.server.transforms.search import BM25SearchTransform, RegexSearchTransform

MCP_ROOT = Path(__file__).resolve().parents[1]  # cognee-mcp/
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

from src.server import TOOL_SEARCH_MAX_RESULTS  # noqa: E402

CATALOG_SIZES = (11, 50, 200, 500)
PINNED = ["remember", "recall", "forget"]
# The value the server actually ships, so the sweep below validates production
# config rather than an arbitrary constant.
TOP_K = TOOL_SEARCH_MAX_RESULTS
K_SWEEP = (3, 5, 10, 15, 25)

# Rough proxy: ~4 chars/token for JSON. Absolute values are indicative; the
# flat-vs-gated ratio is the number that matters and is unit-independent.
CHARS_PER_TOKEN = 4


# --- synthetic catalog ---------------------------------------------------------

ACTIONS = [
    ("create", "Create a new {noun}"),
    ("list", "List every {noun} visible to the current user"),
    ("delete", "Permanently remove a {noun}"),
    ("update", "Modify an existing {noun}"),
    ("describe", "Return full details about a single {noun}"),
]
NOUNS = [
    "dataset",
    "permission",
    "ontology",
    "node_set",
    "pipeline",
    "user",
    "role",
    "credential",
    "graph_projection",
    "data_item",
    "session",
    "tenant",
    "webhook",
    "api_key",
    "notebook",
    "embedding_index",
    "retriever",
    "schedule",
    "export_job",
    "audit_log",
]

# (query, expected tool) — paraphrases that avoid the tool name's exact wording,
# so a name-substring matcher cannot trivially win.
LABELED_QUERIES = [
    ("I want to make a brand new collection of documents", "create_dataset"),
    ("show me everything the current user is allowed to see", "list_permission"),
    ("get rid of a saved credential for good", "delete_credential"),
    ("change the settings on an existing pipeline", "update_pipeline"),
    ("full details about one particular ontology", "describe_ontology"),
    ("wipe out a scheduled job", "delete_schedule"),
    ("enumerate the tenants", "list_tenant"),
    ("set up a fresh api key", "create_api_key"),
    ("modify which nodes are in a group", "update_node_set"),
    ("tell me about a specific audit record", "describe_audit_log"),
]


def mock_tool_specs(count: int) -> list[tuple[str, str, dict]]:
    """Deterministic (name, description, params) triples for `count` mock tools."""
    specs: list[tuple[str, str, dict]] = []
    for noun in NOUNS:
        for action, template in ACTIONS:
            name = f"{action}_{noun}"
            description = (
                f"{template.format(noun=noun.replace('_', ' '))}. "
                f"Operates on the {noun.replace('_', ' ')} resource in the Cognee API."
            )
            params = {"identifier": "The id of the target resource"}
            if action in ("create", "update"):
                params["payload"] = "Fields to write"
            specs.append((name, description, params))
    # Cycle with a suffix if more tools are requested than the vocabulary yields.
    base = list(specs)
    while len(specs) < count:
        name, description, params = base[len(specs) % len(base)]
        specs.append((f"{name}_v{len(specs) // len(base) + 1}", description, params))
    return specs[:count]


def build_server(count: int, transform=None) -> FastMCP:
    """A FastMCP server with `count` mock tools plus the three pinned ones."""
    mcp = FastMCP("benchmark")

    for pinned in PINNED:

        def handler(text: str = "", _name=pinned) -> str:
            """Core memory operation."""
            return _name

        mcp.tool(name=pinned, description=f"Core memory tool {pinned}.")(handler)

    for name, description, params in mock_tool_specs(count):

        def handler(identifier: str = "", payload: str = "", _name=name) -> str:
            return _name

        mcp.tool(name=name, description=description)(handler)

    if transform is not None:
        mcp.add_transform(transform)
    return mcp


# --- the alternative we would have hand-rolled ---------------------------------


class NaiveSubstringSearch:
    """Substring/token-overlap ranking over name + description.

    Stands in for the "v1: lexical scoring, zero deps" implementation the
    original design proposed writing. Scores by how many query tokens appear in
    the tool's text, ties broken by catalog order — no IDF, no length
    normalization. Good enough to beat BM25 on recall at k=5, and beaten by it
    at the k=10 we actually ship; see the module docstring.
    """

    def __init__(self, specs: list[tuple[str, str, dict]]):
        self.docs = [(name, f"{name} {description}".lower()) for name, description, _ in specs]

    def search(self, query: str, top_k: int = TOP_K) -> list[str]:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        scored = []
        for index, (name, text) in enumerate(self.docs):
            score = sum(1 for token in tokens if token in text)
            if score:
                scored.append((-score, index, name))
        scored.sort()
        return [name for _, _, name in scored[:top_k]]


# --- metrics ------------------------------------------------------------------


def recall_and_mrr(results_by_query: dict[str, list[str]]) -> tuple[float, float]:
    """recall@k (did the right tool appear at all) and mean reciprocal rank."""
    hits, reciprocal_ranks = 0, []
    for query, expected in LABELED_QUERIES:
        found = results_by_query[query]
        if expected in found:
            hits += 1
            reciprocal_ranks.append(1 / (found.index(expected) + 1))
        else:
            reciprocal_ranks.append(0.0)
    return hits / len(LABELED_QUERIES), statistics.mean(reciprocal_ranks)


async def list_tools_payload(mcp: FastMCP) -> tuple[int, int]:
    """(tool count, serialized bytes) of what the client receives from tools/list."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    payload = json.dumps(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            for tool in tools
        ]
    )
    return len(tools), len(payload)


async def run_search(mcp: FastMCP, argument: str, queries: list[str]) -> tuple[dict, list[float]]:
    """Execute `queries` through the server's synthetic search tool.

    `argument` is "query" for BM25 and "pattern" for the regex transform.
    """
    results, timings = {}, []
    async with Client(mcp) as client:
        for query in queries:
            started = time.perf_counter()
            result = await client.call_tool("search_tools", {argument: query})
            timings.append((time.perf_counter() - started) * 1000)
            if result.content:
                results[query] = [tool["name"] for tool in json.loads(result.content[0].text)]
            else:
                results[query] = []
    return results, timings


# --- context cost -------------------------------------------------------------


@pytest.mark.parametrize("count", CATALOG_SIZES)
async def test_context_cost_of_gating(count, capsys):
    flat_count, flat_bytes = await list_tools_payload(build_server(count))
    gated_count, gated_bytes = await list_tools_payload(
        build_server(count, BM25SearchTransform(max_results=TOP_K, always_visible=PINNED))
    )

    with capsys.disabled():
        if count == CATALOG_SIZES[0]:
            print(
                f"\n{'catalog':>8} | {'flat tools':>10} {'flat tokens':>12} | "
                f"{'gated tools':>11} {'gated tokens':>12} | {'saved':>7}"
            )
            print("-" * 78)
        print(
            f"{count:>8} | {flat_count:>10} {flat_bytes // CHARS_PER_TOKEN:>12} | "
            f"{gated_count:>11} {gated_bytes // CHARS_PER_TOKEN:>12} | "
            f"{1 - gated_bytes / flat_bytes:>6.1%}"
        )

    # Gated listing is the pinned tools plus search_tools/call_tool, regardless
    # of how large the catalog grows — that is the whole point.
    assert gated_count == len(PINNED) + 2
    assert gated_bytes < flat_bytes
    if count >= 200:
        # An order of magnitude smaller once the catalog is realistically large.
        assert gated_bytes * 10 < flat_bytes


# --- retrieval quality --------------------------------------------------------


def as_regex_pattern(query: str) -> str:
    """What a competent agent would send to RegexSearchTransform.

    Feeding it raw prose scores 0% — it is a pattern matcher, not a ranker — so
    comparing it fairly means giving it the query's content words as alternates.
    """
    tokens = [token for token in query.lower().split() if len(token) > 3]
    return "|".join(tokens)


@pytest.mark.parametrize("count", (50, 500))
async def test_retrieval_quality_versus_hand_rolled(count, capsys):
    queries = [query for query, _ in LABELED_QUERIES]

    bm25_results, bm25_timings = await run_search(
        build_server(count, BM25SearchTransform(max_results=TOP_K, always_visible=PINNED)),
        "query",
        queries,
    )
    regex_raw, regex_timings = await run_search(
        build_server(count, RegexSearchTransform(max_results=TOP_K, always_visible=PINNED)),
        "pattern",
        [as_regex_pattern(query) for query in queries],
    )
    # Re-key by the original query so recall_and_mrr can line results up.
    regex_results = {query: regex_raw[as_regex_pattern(query)] for query in queries}

    naive = NaiveSubstringSearch(mock_tool_specs(count))
    naive_timings = []
    naive_results = {}
    for query in queries:
        started = time.perf_counter()
        naive_results[query] = naive.search(query, TOP_K)
        naive_timings.append((time.perf_counter() - started) * 1000)

    rows = []
    for label, results, timings in (
        ("fastmcp BM25", bm25_results, bm25_timings),
        ("fastmcp regex", regex_results, regex_timings),
        ("hand-rolled", naive_results, naive_timings),
    ):
        recall, mrr = recall_and_mrr(results)
        rows.append((label, recall, mrr, statistics.median(timings), max(timings)))

    with capsys.disabled():
        print(f"\ncatalog = {count} tools, top_k = {TOP_K}, {len(queries)} paraphrase queries")
        print(f"{'strategy':>16} | {f'recall@{TOP_K}':>9} {'MRR':>6} | {'p50 ms':>7} {'max ms':>7}")
        print("-" * 60)
        for label, recall, mrr, p50, worst in rows:
            print(f"{label:>16} | {recall:>8.0%} {mrr:>6.2f} | {p50:>7.2f} {worst:>7.2f}")

    bm25_recall, bm25_mrr = recall_and_mrr(bm25_results)
    naive_recall, naive_mrr = recall_and_mrr(naive_results)

    with capsys.disabled():
        print(
            f"{'':>16}   BM25 vs hand-rolled: "
            f"recall {bm25_recall - naive_recall:+.0%}, MRR {bm25_mrr - naive_mrr:+.2f}"
        )

    # No comparative assertion on purpose: BM25 does *not* dominate the
    # hand-rolled matcher here (see the module docstring), and encoding either
    # direction as a requirement would be asserting an artifact of this fixed
    # query set. What matters is that both clear a usable floor, so a genuine
    # retrieval regression still fails the suite.
    assert bm25_recall >= 0.5, "BM25 lost the target for most queries"
    assert naive_recall >= 0.5, "labeled query set may no longer be solvable"

    # Query cost stays interactive even at 500 tools (generous bound: CI is slow
    # and the first call also builds the index).
    assert statistics.median(bm25_timings) < 250


async def test_recall_versus_max_results(capsys):
    """The sweep that sets TOOL_SEARCH_MAX_RESULTS.

    Recall at a single k conflates the ranker with the window: BM25 trails the
    hand-rolled matcher at k=5 and beats it at k=10, purely because its hits sat
    just outside the narrower window. The shipped value must be at or past the
    knee of this curve.
    """
    queries = [query for query, _ in LABELED_QUERIES]
    naive = NaiveSubstringSearch(mock_tool_specs(500))

    rows = []
    for k in K_SWEEP:
        bm25_results, _ = await run_search(
            build_server(500, BM25SearchTransform(max_results=k, always_visible=PINNED)),
            "query",
            queries,
        )
        bm25_recall, bm25_mrr = recall_and_mrr(bm25_results)
        naive_recall, _ = recall_and_mrr({q: naive.search(q, k) for q in queries})
        rows.append((k, bm25_recall, bm25_mrr, naive_recall))

    by_k = {k: bm25_recall for k, bm25_recall, _, _ in rows}

    with capsys.disabled():
        print(f"\ncatalog = 500 tools — recall@k sweep (shipping k={TOOL_SEARCH_MAX_RESULTS})")
        print(f"{'k':>4} | {'BM25 recall':>11} {'BM25 MRR':>9} | {'hand-rolled':>11}")
        print("-" * 46)
        for k, bm25_recall, bm25_mrr, naive_recall in rows:
            marker = "  <- shipping" if k == TOOL_SEARCH_MAX_RESULTS else ""
            print(f"{k:>4} | {bm25_recall:>10.0%} {bm25_mrr:>9.2f} | {naive_recall:>10.0%}{marker}")

    assert TOOL_SEARCH_MAX_RESULTS in by_k, "shipping k is not covered by the sweep"

    # Recall is monotonic in k (a wider window can only add hits), so the knee is
    # where it stops improving. The shipped k must be no worse than the plateau.
    assert by_k[TOOL_SEARCH_MAX_RESULTS] == max(by_k.values()), (
        f"k={TOOL_SEARCH_MAX_RESULTS} leaves recall on the table: {by_k}"
    )
    # ...and strictly better than the narrow window that made the ranker look bad.
    assert by_k[TOOL_SEARCH_MAX_RESULTS] > by_k[3]


async def test_bm25_index_is_built_once_and_reused(capsys):
    """The index builds lazily on first search and is reused after, so only the
    first query pays for it. Guards against a per-call rebuild regression."""
    queries = [query for query, _ in LABELED_QUERIES] * 3
    _, timings = await run_search(
        build_server(500, BM25SearchTransform(max_results=TOP_K, always_visible=PINNED)),
        "query",
        queries,
    )

    first, rest = timings[0], timings[1:]
    with capsys.disabled():
        print(
            f"\n500 tools: first query {first:.2f} ms (index build), "
            f"subsequent p50 {statistics.median(rest):.2f} ms"
        )

    assert statistics.median(rest) <= first
