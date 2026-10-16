"""E2E test: code ingestion -> enola code graph -> SearchType.CODE (SDK-395).

Runs the REAL enola binary (the enola-cli wheel included with cognee) over
a small pinned repository that ships with the test suite, against the default
embedded databases (Ladybug graph + SQLite), with NO LLM key. The code graph
itself calls neither the LLM nor an embedding model; the docstring chunks the
code routes store next to it (CODE_GRAPH_DOCS, on by default) are embedded,
on the local fastembed model when no provider is configured -- see
_code_docs_enabled for the two ways to run this test. It covers what the unit
tests in tests/unit/tasks/code_graph mock away: the enola output contract,
writing typed code nodes and dynamic relation edges into a real graph store,
reading them back through the CODE retriever with dataset scoping, idempotent
re-ingestion, and teardown.

Verifies:
- remember(content_type="code") ingests the repo and reports a completed run
- the code graph holds ONLY the typed enola models (CodeRepository, CodeModule,
  CodeSymbol, ...) — no generic Node / Entity / TextSummary, and no chunks
  other than the docstring layer below
- with CODE_GRAPH_DOCS on, every ingestion also stores one CodeDocument per
  documented fixture file and one DocumentChunk per docstring (the fixture has
  module docstrings only), each linked to its CodeModule by a `describes`
  edge; an unchanged re-ingestion adds none, and forget(data_id=...) removes
  them along with the repository's code graph
- known files, symbols, modules, dependencies and typed relations
  (calls / imports / declares / part_of) are present, with resolved endpoints
- SearchType.CODE known-answer queries (query_facts, find_path,
  impact_analysis) through both search() and recall()
- re-ingesting the unchanged repo is a no-op (same node/edge counts, same
  snapshot id)
- every remembered repository is one Data row whose id the result carries, and
  forget(data_id=...) removes exactly that repository's graph (SDK-783)
- add(<project dir>) + cognify() routes the repo down the CODE_REPO route and
  produces the same typed graph
- forget(everything=True) leaves no datasets or graph state behind

The fixture lives in tests/test_data/code_repo_fixture and is COPIED before
ingestion because enola writes its .enola/ snapshot into the scanned tree.
"""

import asyncio
import importlib.metadata
import json
import os
import pathlib
import shutil
import tempfile
from uuid import UUID

import cognee
from cognee import SearchType
from cognee.api.v1.datasets.datasets import datasets
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods.get_dataset_processing_status import (
    get_dataset_processing_status,
)
from cognee.modules.retrieval.code_retriever import CODE_NODE_TYPES
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

# The enola version cognee runs is the enola-cli wheel pinned in pyproject.toml;
# the known answers below are pinned to it.
ENOLA_PINNED_VERSION = importlib.metadata.version("enola-cli")

logger = get_logger()

TESTS_DIR = pathlib.Path(__file__).parent
FIXTURE_DIR = TESTS_DIR / "test_data" / "code_repo_fixture"

REMEMBER_DATASET = "code_graph_e2e_remember"
COGNIFY_DATASET = "code_graph_e2e_cognify"

# --- Known answers, pinned to the fixture + ENOLA_PINNED_VERSION ------------
# If enola is bumped and these change, update them deliberately. Last
# re-verified against enola 0.4.30: it emits the README as document/section
# symbols, an `extraction` coverage fact, the pyproject's declared package as
# `pkg:pypi/requests`, and (since 0.4.14) a module-level `module-edge:` dependency
# fact per resolved package import. File-level import facts such as
# `main -> inventory.store` now carry the imported file path as their target and
# no resolved module id; the resolved `imports` edge lives on the module-edge fact.

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
# would show up as one of these. Their presence in the code graph means the
# code did NOT take the enola route, or the route emitted generic models. The
# docstring layer the code routes store themselves (CodeDocument nodes and the
# DocumentChunks cut as code docstrings) is split off before this check and
# asserted on its own (_assert_code_docs).
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

# The docstring layer: cognee/tasks/code_docs, run by the code routes' docs tail.
CODE_DOCSTRING_CUT_TYPE = "code_docstring"

# The fixture's docstrings are module docstrings only (no function or class has
# one), so each documented file yields exactly one chunk, headed "<file>:1" and
# linked to the CodeModule of its directory (the repo root is ".").
EXPECTED_DOCSTRING_CHUNKS = {
    "main.py:1": ".",
    "inventory/pricing.py:1": "inventory",
    "inventory/store.py:1": "inventory",
}


def _code_docs_enabled() -> bool:
    """Whether this run stores, and so asserts, the docstring layer: the routes' own setting.

    Two ways to run this test, picked by the environment:

    - Docstrings ON (CODE_GRAPH_DOCS unset or true, the default): the chunks are
      embedded on the local fastembed model, still without any key. Keyless, the
      remember() steps would reach that model through the keyless fallback on
      their own, but the cognify() step sets COGNEE_SKIP_CONNECTION_TEST (see
      step 4), which also switches the fallback off (keyless_local_defaults_apply)
      and would leave the CODE_REPO route's tail with the OpenAI default and no
      key. An explicit setting is honoured on every path, so main() pins the same
      local model (EMBEDDING_PROVIDER=fastembed, BAAI/bge-small-en-v1.5, 384
      dimensions, downloaded on first use) unless the environment already
      configures an embedding setting (provider, model, key or endpoint), which
      then embeds the docstrings itself. _assert_code_docs then runs after every
      ingestion.
    - Docstrings OFF (CODE_GRAPH_DOCS=false in the environment before this module
      runs): the graph holds the code graph alone, which this test asserts, and the
      docstring round trip stays covered by
      cognee/tests/unit/tasks/code_docs/test_code_docs_round_trip.py.
    """
    return get_cognify_config().code_graph_docs


# The settings embedding_settings_configured() reads: any of them set means the
# environment chose its embedder, and the pin below stays out of its way.
_EMBEDDING_SETTINGS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_ENDPOINT",
)


def _pin_local_embeddings() -> None:
    """Embed the docstring layer on the local fastembed model for the whole run.

    See _code_docs_enabled: the keyless fallback would pick this model for the
    remember() steps, and step 4's COGNEE_SKIP_CONNECTION_TEST switches the
    fallback off, so the same model is named explicitly -- provider, model and
    dimensions together, and only when the environment configures none of the
    embedding settings (a model or key named without a provider must not be
    paired with fastembed). No key is involved.
    """
    if any(os.environ.get(name) for name in _EMBEDDING_SETTINGS):
        return
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["EMBEDDING_MODEL"] = "BAAI/bge-small-en-v1.5"
    os.environ["EMBEDDING_DIMENSIONS"] = "384"


# --- Helpers -----------------------------------------------------------------


def _copy_fixture(destination_root: str, name: str = FIXTURE_DIR.name) -> pathlib.Path:
    """Copy the pinned fixture; enola writes .enola/ into the tree it scans.

    The directory name is the repository name, which scopes the fact node ids:
    two copies under the same name would share one set of nodes.
    """
    target = pathlib.Path(destination_root) / name
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


def _split_code_docs(nodes: dict, edges: list) -> tuple[dict, list, dict, list]:
    """(code nodes, code edges, docstring nodes, docstring edges) of a stored graph."""
    docs_nodes = {
        node_id: p
        for node_id, p in nodes.items()
        if p.get("type") == "CodeDocument"
        or (p.get("type") == "DocumentChunk" and p.get("cut_type") == CODE_DOCSTRING_CUT_TYPE)
    }
    code_nodes = {node_id: p for node_id, p in nodes.items() if node_id not in docs_nodes}
    docs_edges = [e for e in edges if e[0] in docs_nodes or e[1] in docs_nodes]
    code_edges = [e for e in edges if e[0] not in docs_nodes and e[1] not in docs_nodes]
    return code_nodes, code_edges, docs_nodes, docs_edges


def _assert_code_docs(code_nodes: dict, docs_nodes: dict, docs_edges: list, repo_name: str) -> None:
    """The docstring layer of a single-repository graph (CODE_GRAPH_DOCS on)."""
    documents = _by_type(docs_nodes, "CodeDocument")
    chunks = _by_type(docs_nodes, "DocumentChunk")
    assert documents and chunks, (
        "CODE_GRAPH_DOCS is on but the code route stored no docstrings: "
        f"{len(documents)} CodeDocument(s), {len(chunks)} DocumentChunk(s)"
    )
    # Only this repository's: after forget(data_id=...) of another repository,
    # its documents and chunks are gone along with its code graph.
    foreign = sorted({p.get("repo") for p in documents.values()} - {repo_name})
    assert not foreign, f"CodeDocuments of other repositories remain: {foreign}"
    documented = {p.get("name") for p in documents.values()}
    assert EXPECTED_FILES <= documented, (
        f"Documented files without a CodeDocument: {EXPECTED_FILES - documented}"
    )

    headers = {node_id: (p.get("text") or "").split("\n", 1)[0] for node_id, p in chunks.items()}
    missing = set(EXPECTED_DOCSTRING_CHUNKS) - set(headers.values())
    assert not missing, (
        f"Docstring chunks missing (chunk text starts with file:line): {sorted(missing)}"
    )

    module_names = {
        node_id: p.get("name") for node_id, p in _by_type(code_nodes, "CodeModule").items()
    }
    describes = {s: t for s, t, rel, _ in docs_edges if rel == "describes"}
    is_part_of = {s: t for s, t, rel, _ in docs_edges if rel == "is_part_of"}
    for chunk_id, header in headers.items():
        assert is_part_of.get(chunk_id) in documents, (
            f"Docstring chunk {header!r} is not linked to its CodeDocument (is_part_of)"
        )
        expected_module = EXPECTED_DOCSTRING_CHUNKS.get(header)
        if expected_module is None:
            continue
        target = describes.get(chunk_id)
        assert target is not None, f"Docstring chunk {header!r} has no 'describes' edge"
        assert module_names.get(target) == expected_module, (
            f"Docstring chunk {header!r} describes {module_names.get(target, target)!r}, "
            f"expected CodeModule {expected_module!r}"
        )


def _assert_typed_code_graph(nodes: dict, edges: list, repo_name: str) -> None:
    """Structural + known-entity assertions shared by both ingestion routes."""
    assert nodes, "Graph is empty — the code graph pipeline stored nothing"

    # 0. Every edge endpoint exists -- docstring edges included, so a swept or
    #    forgotten repository leaves no `describes` edge behind.
    dangling = [e for e in edges if e[0] not in nodes or e[1] not in nodes]
    assert not dangling, (
        f"{len(dangling)} edge(s) reference nodes that do not exist: {dangling[:3]}"
    )

    # The docstring layer is asserted on its own; the typed-graph checks below
    # see the code graph alone.
    code_nodes, code_edges, docs_nodes, docs_edges = _split_code_docs(nodes, edges)
    if _code_docs_enabled():
        _assert_code_docs(code_nodes, docs_nodes, docs_edges, repo_name)
    else:
        assert not docs_nodes, (
            "CODE_GRAPH_DOCS=false must leave the code graph alone, found "
            f"{len(docs_nodes)} docstring node(s)"
        )
    nodes, edges = code_nodes, code_edges

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

    # 4. Relations: known calls/declares/part_of exist (endpoints were checked in 0).
    name_of = {node_id: p.get("name") for node_id, p in nodes.items()}
    calls = {(name_of[s], name_of[t]) for s, t, rel, _ in edges if rel == "calls"}
    assert EXPECTED_CALLS <= calls, f"Missing 'calls' edges: {sorted(EXPECTED_CALLS - calls)}"

    declares = {(name_of[s], name_of[t]) for s, t, rel, _ in edges if rel == "declares"}
    assert ("inventory/pricing.line_total", "inventory") in declares, (
        "Missing 'declares' edge from inventory/pricing.line_total to module 'inventory'"
    )

    imports = {(name_of[s], name_of[t]) for s, t, rel, _ in edges if rel == "imports"}
    # enola resolves the package import from main.py to the top-level module
    # `inventory` on a module-level dependency fact (see the known-answers note).
    assert ("module-edge: . -> inventory", "inventory") in imports, (
        f"Missing 'imports' edge from 'module-edge: . -> inventory' to module 'inventory'; "
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
    if _code_docs_enabled():
        _pin_local_embeddings()

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
        first_data_id = result.items[0].get("id")
        assert first_data_id, f"remember() returned no data id for the repository: {result.items}"

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
        assert result.items[0].get("id") == first_data_id, (
            "Re-remembering the same repository minted a new data id"
        )
        await _assert_code_search(REMEMBER_DATASET)

        # --- 3b. One Data row per repo; forget(data_id) drops only it (SDK-783)
        user = await get_default_user()
        remember_dataset = next(
            ds for ds in await datasets.list_datasets(user=user) if ds.name == REMEMBER_DATASET
        )

        # A changed repository refreshes its row in place: same id, new manifest.
        (repo_data_row,) = await datasets.list_data(remember_dataset.id, user=user)
        (repo_path / "inventory" / "audit.py").write_text("def audit():\n    return True\n")
        result = await cognee.remember(
            str(repo_path), dataset_name=REMEMBER_DATASET, content_type="code"
        )
        assert result.status == "completed", result.error
        assert result.items[0].get("id") == first_data_id, "A changed repo minted a new data id"
        (refreshed_row,) = await datasets.list_data(remember_dataset.id, user=user)
        assert refreshed_row.content_hash != repo_data_row.content_hash, (
            "Re-remembering a changed repository kept its old manifest"
        )
        assert (
            _system_metadata(refreshed_row)["file_count"]
            == _system_metadata(repo_data_row)["file_count"] + 1
        )
        async with set_database_global_context_variables(remember_dataset.id, user.id):
            nodes, _edges = await _graph_snapshot()
        assert "inventory/audit.audit" in _named(_by_type(nodes, "CodeSymbol")), (
            "The changed repository refreshed its row but not its graph"
        )

        # An unchanged repository whose re-sync fails keeps its stamps: its graph
        # is still built, so a later cognify() of the dataset must not pick the
        # row up again (it would rerun enola, or fail when the clone is gone).
        import cognee.modules.run_custom_pipeline as custom_pipeline_module

        real_run_custom_pipeline = custom_pipeline_module.run_custom_pipeline

        async def _failing_code_graph_run(*_args, **_kwargs):
            raise RuntimeError("simulated code graph failure")

        custom_pipeline_module.run_custom_pipeline = _failing_code_graph_run
        try:
            result = await cognee.remember(
                str(repo_path),
                dataset_name=REMEMBER_DATASET,
                content_type="code",
                raise_on_error=False,
            )
        finally:
            custom_pipeline_module.run_custom_pipeline = real_run_custom_pipeline
        assert result.status == "errored", result
        assert result.items[0].get("id") == first_data_id, result.items
        processing = await get_dataset_processing_status(remember_dataset.id)
        assert processing["pending"] == 0, (
            f"A failed re-sync of an unchanged repository dropped its cognify stamp: {processing}"
        )

        second_repo_path = _copy_fixture(os.path.join(scratch_root, "second"), name="second_repo")
        result = await cognee.remember(
            str(second_repo_path), dataset_name=REMEMBER_DATASET, content_type="code"
        )
        assert result.status == "completed", result.error
        second_data_id = result.items[0].get("id")

        data_rows = await datasets.list_data(remember_dataset.id, user=user)
        assert {str(row.id) for row in data_rows} == {first_data_id, second_data_id}, (
            f"Expected one Data row per remembered repository, got {[row.name for row in data_rows]}"
        )
        assert all(_system_metadata(row).get("source") == "code_repo" for row in data_rows)
        # Stamped as cognified (a later cognify() of the dataset has nothing to
        # redo) and in the code graph pipeline's own per-item slot.
        for pipeline_name in ("cognify_pipeline", "code_graph_pipeline"):
            processing = await get_dataset_processing_status(remember_dataset.id, pipeline_name)
            assert processing["pending"] == 0, (
                f"Repositories left pending for {pipeline_name}: {processing}"
            )

        await cognee.forget(data_id=UUID(first_data_id), dataset_id=remember_dataset.id)

        async with set_database_global_context_variables(remember_dataset.id, user.id):
            nodes, edges = await _graph_snapshot()
        # Exactly one repository left, fully linked, no dangling edges: the
        # forgotten repository's nodes and edges are all gone.
        _assert_typed_code_graph(nodes, edges, repo_name=second_repo_path.name)
        remaining_rows = await datasets.list_data(remember_dataset.id, user=user)
        assert [str(row.id) for row in remaining_rows] == [second_data_id]

        # --- 4. add(<project dir>) + cognify(): the CODE_REPO route -----------
        # cognify() runs the LLM/embedding connection test unconditionally,
        # even when every item routes to an LLM-free task list. Skip it via
        # the documented switch — set only NOW so the remember() steps above
        # still prove that remember(content_type="code") is keyless on its own.
        # The switch also disables the keyless fastembed fallback, which is why
        # main() pinned the local embedding model explicitly (see
        # _code_docs_enabled) rather than relying on the fallback here.
        os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
        cognify_repo_path = _copy_fixture(os.path.join(scratch_root, "cognify"))
        await cognee.add(str(cognify_repo_path), dataset_name=COGNIFY_DATASET)

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

        # Re-adding the changed project reaches ingestion's content check, so the
        # next cognify() rebuilds its graph instead of skipping it as done.
        (cognify_repo_path / "inventory" / "audit.py").write_text("def audit():\n    return True\n")
        await cognee.add(str(cognify_repo_path), dataset_name=COGNIFY_DATASET)
        await cognee.cognify(datasets=[COGNIFY_DATASET])
        async with set_database_global_context_variables(cognify_dataset.id, user.id):
            nodes, _edges = await _graph_snapshot()
        assert "inventory/audit.audit" in _named(_by_type(nodes, "CodeSymbol")), (
            "add() of a changed code project kept its old manifest and graph"
        )

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
