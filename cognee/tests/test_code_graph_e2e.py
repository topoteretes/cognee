"""E2E test: code ingestion -> enola code graph -> SearchType.CODE (SDK-395).

Runs the REAL enola binary (pinned release, auto-installed on first use) over
a small pinned repository that ships with the test suite, against the default
embedded databases (Ladybug graph + SQLite), with NO LLM or embedding
configuration. It covers what the unit tests in tests/unit/tasks/code_graph
mock away: the enola output contract, writing typed code nodes and dynamic
relation edges into a real graph store, reading them back through the CODE
retriever with dataset scoping, idempotent re-ingestion, and teardown.

Verifies:
- remember(content_type="code") ingests the repo and reports a completed run
- the graph holds ONLY the typed enola models (CodeRepository, CodeModule,
  CodeSymbol, ...) — no generic Node / Entity / DocumentChunk / TextSummary
- known files, symbols, modules, dependencies and typed relations
  (calls / imports / declares / part_of) are present, with resolved endpoints
- SearchType.CODE known-answer queries (query_facts, find_path,
  impact_analysis) through both search() and recall()
- re-ingesting the unchanged repo is a no-op (same node/edge counts, same
  snapshot id)
- add(<project dir>) + cognify() routes the repo down the CODE_REPO route and
  produces the same typed graph
- forget(everything=True) leaves no datasets or graph state behind

The fixture lives in tests/test_data/code_repo_fixture and is COPIED before
ingestion because enola writes its .enola/ snapshot into the scanned tree.
"""

import asyncio
import json
import os
import pathlib
import shutil
import tempfile

import cognee
from cognee import SearchType
from cognee.api.v1.datasets.datasets import datasets
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.code_retriever import CODE_NODE_TYPES
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph import ENOLA_PINNED_VERSION

logger = get_logger()

TESTS_DIR = pathlib.Path(__file__).parent
FIXTURE_DIR = TESTS_DIR / "test_data" / "code_repo_fixture"

REMEMBER_DATASET = "code_graph_e2e_remember"
COGNIFY_DATASET = "code_graph_e2e_cognify"

# --- Known answers, pinned to the fixture + ENOLA_PINNED_VERSION ------------
# If enola is bumped and these change, update them deliberately. Last
# re-verified against enola 0.4.12 (which additionally emits the README as
# document/section symbols, an `extraction` coverage fact, and the
# pyproject's declared package as `pkg:pypi/requests`).

EXPECTED_MODULES = {".", "inventory"}

EXPECTED_SYMBOLS = {
    "main.main": "function",
    "inventory/pricing.line_total": "function",
    "inventory/pricing.compute_total": "function",
    "inventory/pricing.apply_discount": "function",
    "inventory/store.InventoryStore": "class",
    "inventory/store.InventoryStore.total": "method",
    "inventory/store.InventoryStore.discounted_total": "method",
}

EXPECTED_FILES = {"main.py", "inventory/pricing.py", "inventory/store.py"}

# A symbol whose file this is, per file — the "known files" check goes through
# the CodeSymbol.file_path field rather than a file node (enola has no such kind).
EXPECTED_DEPENDENCY_NAMES = {
    "inventory/store -> requests",
    "inventory/pricing -> math",
    "inventory/store -> inventory.pricing",
}

EXPECTED_CALLS = {
    ("inventory/pricing.apply_discount", "inventory/pricing.compute_total"),
    ("inventory/pricing.compute_total", "inventory/pricing.line_total"),
    ("inventory/store.InventoryStore.total", "inventory/pricing.compute_total"),
    ("inventory/store.InventoryStore.discounted_total", "inventory/pricing.apply_discount"),
    ("main.main", "inventory/store.InventoryStore.discounted_total"),
}

# main.main -> discounted_total -> apply_discount -> compute_total -> line_total
PATH_SOURCE = "main.main"
PATH_TARGET = "inventory/pricing.line_total"
PATH_HOPS = 4

# Anything ingested by the standard (LLM) cognify pipeline or the DLT route
# would show up as one of these. Their presence means the code did NOT take the
# enola route, or the route emitted generic models.
FORBIDDEN_NODE_TYPES = {
    "Node",
    "Entity",
    "EntityType",
    "Document",
    "TextDocument",
    "CodeFileDocument",
    "DocumentChunk",
    "TextSummary",
    "CodeFile",
    "FunctionDefinition",
    "ClassDefinition",
    "CodePart",
}

ALLOWED_NODE_TYPES = set(CODE_NODE_TYPES) | {"CodeRepository"}


# --- Helpers -----------------------------------------------------------------


def _copy_fixture(destination_root: str) -> pathlib.Path:
    """Copy the pinned fixture; enola writes .enola/ into the tree it scans."""
    target = pathlib.Path(destination_root) / FIXTURE_DIR.name
    shutil.copytree(FIXTURE_DIR, target)
    return target


async def _graph_snapshot():
    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    # Ladybug fabricates self-referential edges for an edge-less graph; ignore them.
    edges = [edge for edge in edges if edge[0] != edge[1]]
    return {node_id: props for node_id, props in nodes}, edges


def _by_type(nodes: dict, node_type: str) -> dict:
    return {node_id: p for node_id, p in nodes.items() if p.get("type") == node_type}


def _named(nodes: dict) -> dict:
    """name -> properties, for the typed code nodes (names are unique per kind here)."""
    return {p.get("name"): p for p in nodes.values()}


def _assert_typed_code_graph(nodes: dict, edges: list, repo_name: str) -> None:
    """Structural + known-entity assertions shared by both ingestion routes."""
    assert nodes, "Graph is empty — the code graph pipeline stored nothing"

    # 1. Only the typed enola models, never generic/document nodes.
    types_present = {p.get("type") for p in nodes.values()}
    forbidden = types_present & FORBIDDEN_NODE_TYPES
    assert not forbidden, (
        f"Generic/document node types found in the code graph: {sorted(forbidden)}. "
        "The repository did not take the enola code-graph route or the route "
        "emitted untyped models."
    )
    unexpected = types_present - ALLOWED_NODE_TYPES
    assert not unexpected, f"Unexpected node types in code graph: {sorted(unexpected)}"

    # 2. Exactly one repository node, stamped with the snapshot identity.
    repositories = _by_type(nodes, "CodeRepository")
    assert len(repositories) == 1, (
        f"Expected exactly 1 CodeRepository node, got {len(repositories)}: "
        f"{[p.get('name') for p in repositories.values()]}"
    )
    repository = next(iter(repositories.values()))
    assert repository.get("name") == repo_name, (
        f"CodeRepository name {repository.get('name')!r} != {repo_name!r}"
    )
    assert repository.get("last_snapshot_id"), (
        "CodeRepository.last_snapshot_id was not stamped — add_code_graph_edges did not complete"
    )
    receipt = repository.get("last_receipt")
    if isinstance(receipt, str):
        receipt = json.loads(receipt)
    assert isinstance(receipt, dict) and receipt.get("format_version") == 1, (
        f"CodeRepository.last_receipt should carry the snapshot's receipt projection, got {receipt!r}"
    )
    assert receipt.get("enola_version") == ENOLA_PINNED_VERSION, (
        f"Graph was built by enola {receipt.get('enola_version')!r}, pinned {ENOLA_PINNED_VERSION!r}"
    )

    # 3. Known modules, symbols (with symbol_kind), files, dependencies.
    modules = {p.get("name") for p in _by_type(nodes, "CodeModule").values()}
    assert EXPECTED_MODULES <= modules, f"Missing CodeModule nodes: {EXPECTED_MODULES - modules}"

    symbols = _named(_by_type(nodes, "CodeSymbol"))
    missing_symbols = set(EXPECTED_SYMBOLS) - set(symbols)
    assert not missing_symbols, f"Missing CodeSymbol nodes: {sorted(missing_symbols)}"
    for name, expected_kind in EXPECTED_SYMBOLS.items():
        actual_kind = symbols[name].get("symbol_kind")
        assert actual_kind == expected_kind, (
            f"CodeSymbol {name!r}: symbol_kind {actual_kind!r} != {expected_kind!r}"
        )

    files = {p.get("file_path") for p in symbols.values()}
    assert EXPECTED_FILES <= files, (
        f"Known files missing from CodeSymbol.file_path: {EXPECTED_FILES - files}"
    )

    dependencies = {p.get("name") for p in _by_type(nodes, "ExternalDependency").values()}
    assert EXPECTED_DEPENDENCY_NAMES <= dependencies, (
        f"Missing ExternalDependency nodes: {EXPECTED_DEPENDENCY_NAMES - dependencies}"
    )

    # 4. Relations: every endpoint resolves to a node; known calls/declares/part_of exist.
    dangling = [e for e in edges if e[0] not in nodes or e[1] not in nodes]
    assert not dangling, (
        f"{len(dangling)} edge(s) reference nodes that do not exist: {dangling[:3]}"
    )

    name_of = {node_id: p.get("name") for node_id, p in nodes.items()}
    calls = {(name_of[s], name_of[t]) for s, t, rel, _ in edges if rel == "calls"}
    assert EXPECTED_CALLS <= calls, f"Missing 'calls' edges: {sorted(EXPECTED_CALLS - calls)}"

    declares = {(name_of[s], name_of[t]) for s, t, rel, _ in edges if rel == "declares"}
    assert ("inventory/pricing.line_total", "inventory") in declares, (
        "Missing 'declares' edge from inventory/pricing.line_total to module 'inventory'"
    )

    imports = {(name_of[s], name_of[t]) for s, t, rel, _ in edges if rel == "imports"}
    # enola resolves this import's target to the top-level package module.
    assert ("main -> inventory.store", "inventory") in imports, (
        f"Missing 'imports' edge from 'main -> inventory.store' to module 'inventory'; "
        f"imports present: {sorted(imports)}"
    )

    repo_id = next(iter(repositories))
    part_of_targets = {t for s, t, rel, _ in edges if rel == "part_of"}
    assert repo_id in part_of_targets, "No 'part_of' edges point at the CodeRepository node"
    entities_without_repo = [
        name_of[node_id]
        for node_id in nodes
        if node_id != repo_id
        and not any(s == node_id and rel == "part_of" for s, _, rel, _ in edges)
    ]
    assert not entities_without_repo, (
        f"Code entities not linked to their repository via part_of: {entities_without_repo[:5]}"
    )


async def _assert_code_search(dataset_name: str) -> None:
    """Known-answer SearchType.CODE queries: search() and recall() surfaces."""
    # query_facts: list symbols, expect the known ones.
    result = await cognee.search(
        query_type=SearchType.CODE,
        query_text="",
        datasets=[dataset_name],
        code_query={"operation": "query_facts", "kinds": ["symbol"], "limit": 100},
    )
    payload = _single_code_payload(result, dataset_name)
    assert payload["operation"] == "query_facts", payload
    fact_names = {fact["name"] for fact in payload["facts"]}
    assert set(EXPECTED_SYMBOLS) <= fact_names, (
        f"query_facts is missing symbols: {set(EXPECTED_SYMBOLS) - fact_names}"
    )
    for fact in payload["facts"]:
        assert fact["kind"] == "symbol" and fact["type"] == "CodeSymbol", fact

    # find_path: the fixture's 4-hop call chain.
    result = await cognee.search(
        query_type=SearchType.CODE,
        query_text=PATH_SOURCE,
        datasets=[dataset_name],
        code_query={
            "operation": "find_path",
            "source": PATH_SOURCE,
            "target": PATH_TARGET,
            "relation_types": ["calls"],
        },
    )
    payload = _single_code_payload(result, dataset_name)
    assert payload["found"] is True, f"No call path from {PATH_SOURCE} to {PATH_TARGET}: {payload}"
    path_names = [fact["name"] for fact in payload["path"]]
    assert path_names[0] == PATH_SOURCE and path_names[-1] == PATH_TARGET, path_names
    assert len(payload["edges"]) == PATH_HOPS, (
        f"Expected a {PATH_HOPS}-hop call path, got {len(payload['edges'])}: {path_names}"
    )
    assert all(edge["type"] == "calls" for edge in payload["edges"]), payload["edges"]

    # impact_analysis: who depends (transitively) on line_total.
    result = await cognee.search(
        query_type=SearchType.CODE,
        query_text=PATH_TARGET,
        datasets=[dataset_name],
        code_query={"operation": "impact_analysis", "target": PATH_TARGET, "max_depth": 5},
    )
    payload = _single_code_payload(result, dataset_name)
    affected = {node["name"] for level in payload["by_depth"].values() for node in level}
    expected_affected = {
        "inventory/pricing.compute_total",
        "inventory/pricing.apply_discount",
        "inventory/store.InventoryStore.total",
        "inventory/store.InventoryStore.discounted_total",
        "main.main",
    }
    assert expected_affected <= affected, (
        f"impact_analysis of {PATH_TARGET} is missing dependents: {expected_affected - affected}"
    )

    # recall(): the memory-API surface reaches the same retriever.
    recall_result = await cognee.recall(
        PATH_SOURCE,
        query_type=SearchType.CODE,
        datasets=[dataset_name],
        # recall() only honours code_query when the code scope is explicit.
        scope=["code"],
        code_query={"operation": "explore", "name": PATH_SOURCE, "max_depth": 1},
    )
    recall_text = str(recall_result)
    assert PATH_SOURCE in recall_text and "InventoryStore.discounted_total" in recall_text, (
        f"recall(query_type=CODE) did not return the explore neighbourhood of {PATH_SOURCE}: "
        f"{recall_text[:500]}"
    )


def _system_metadata(row) -> dict:
    metadata = getattr(row, "system_metadata", None)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata or {}


def _single_code_payload(result, dataset_name: str):
    """search() returns one {dataset_id, dataset_name, search_result} per dataset; we search one."""
    assert isinstance(result, list) and len(result) == 1, (
        f"Unexpected CODE search shape: {str(result)[:500]}"
    )
    entry = result[0]
    assert isinstance(entry, dict), f"CODE search entry is not a dict: {type(entry)}"
    assert entry.get("dataset_name") == dataset_name, entry.get("dataset_name")
    payload = entry.get("search_result")
    assert isinstance(payload, dict), f"CODE payload is not a dict: {type(payload)}"
    return payload


# --- Scenario ----------------------------------------------------------------


async def main():
    data_directory_path = str((TESTS_DIR / ".data_storage" / "test_code_graph_e2e").resolve())
    cognee_directory_path = str((TESTS_DIR / ".cognee_system" / "test_code_graph_e2e").resolve())
    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    # Enola output must not depend on a previous run's snapshot.
    scratch_root = tempfile.mkdtemp(prefix="cognee_code_graph_e2e_")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    try:
        logger.info("enola pinned version: %s", ENOLA_PINNED_VERSION)

        # --- 1. remember(content_type="code") ---------------------------------
        repo_path = _copy_fixture(os.path.join(scratch_root, "remember"))
        result = await cognee.remember(
            str(repo_path), dataset_name=REMEMBER_DATASET, content_type="code"
        )
        assert result.status == "completed", (
            f"remember() did not complete: {result.status} {result.error}"
        )
        assert result.items_processed == 1 and result.items[0]["kind"] == "code_repository", (
            result.items
        )
        assert result.pipeline_run_id, "remember() reported no pipeline_run_id"
        assert (repo_path / ".enola" / "facts.jsonl").is_file(), "enola snapshot was not written"

        nodes, edges = await _graph_snapshot()
        _assert_typed_code_graph(nodes, edges, repo_name=repo_path.name)
        first_counts = (len(nodes), len(edges))
        first_snapshot_id = next(iter(_by_type(nodes, "CodeRepository").values()))[
            "last_snapshot_id"
        ]
        logger.info("First ingestion: %d nodes, %d edges", *first_counts)

        # --- 2. Known-answer CODE searches ------------------------------------
        await _assert_code_search(REMEMBER_DATASET)

        # --- 3. Re-ingest the unchanged repo: must be a no-op -----------------
        result = await cognee.remember(
            str(repo_path), dataset_name=REMEMBER_DATASET, content_type="code"
        )
        assert result.status == "completed", result.error
        nodes, edges = await _graph_snapshot()
        _assert_typed_code_graph(nodes, edges, repo_name=repo_path.name)
        second_counts = (len(nodes), len(edges))
        assert second_counts == first_counts, (
            f"Re-ingesting an unchanged repo changed the graph: {first_counts} -> {second_counts}"
        )
        second_snapshot_id = next(iter(_by_type(nodes, "CodeRepository").values()))[
            "last_snapshot_id"
        ]
        assert second_snapshot_id == first_snapshot_id, (
            "Snapshot id changed although the repository did not"
        )
        await _assert_code_search(REMEMBER_DATASET)

        # --- 4. add(<project dir>) + cognify(): the CODE_REPO route -----------
        # cognify() runs the LLM/embedding connection test unconditionally,
        # even when every item routes to an LLM-free task list. Skip it via
        # the documented switch — set only NOW so the remember() steps above
        # still prove that remember(content_type="code") is keyless on its own.
        os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
        cognify_repo_path = _copy_fixture(os.path.join(scratch_root, "cognify"))
        await cognee.add(str(cognify_repo_path), dataset_name=COGNIFY_DATASET)

        user = await get_default_user()
        cognify_dataset = next(
            ds for ds in await datasets.list_datasets(user=user) if ds.name == COGNIFY_DATASET
        )
        data_rows = await datasets.list_data(cognify_dataset.id, user=user)
        # The cognify router keys on system_metadata.source, not on the name.
        repo_items = [
            row for row in data_rows if _system_metadata(row).get("source") == "code_repo"
        ]
        assert len(repo_items) == 1, (
            f"Expected exactly one code_repo manifest Data record, got "
            f"{[(row.name, _system_metadata(row).get('source')) for row in data_rows]}"
        )
        # Keyless: the README document is excluded (needs an LLM), so the repo
        # manifest must be the ONLY record — nothing may reach the LLM pipeline.
        assert len(data_rows) == 1, (
            f"Unexpected extra Data records for a keyless code-project add: "
            f"{[row.name for row in data_rows]}"
        )

        await cognee.cognify(datasets=[COGNIFY_DATASET])

        nodes, edges = await _graph_snapshot()
        _assert_typed_code_graph(nodes, edges, repo_name=cognify_repo_path.name)
        await _assert_code_search(COGNIFY_DATASET)

        # --- 5. Teardown leaves nothing behind --------------------------------
        await cognee.forget(everything=True)

        remaining = await datasets.list_datasets(user=user)
        assert remaining == [], (
            f"forget(everything=True) left datasets: {[d.name for d in remaining]}"
        )

        print("\n✅ Code graph e2e test passed")
        print(
            f"   enola {ENOLA_PINNED_VERSION}; graph: {first_counts[0]} nodes, {first_counts[1]} edges"
        )
    finally:
        # Belt and braces: prune whatever the assertions above did not reach,
        # and remove the scratch copies (including their .enola/ snapshots).
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        shutil.rmtree(scratch_root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
