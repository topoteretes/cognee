from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.modules.retrieval.code_coverage import repository_coverage
from cognee.tasks.code_graph.extract_code_graph import build_code_graph_edges, fact_node_id
from cognee.tasks.code_graph.models import CodeSymbol
from cognee.tasks.code_graph.symbol_migration import (
    SymbolMigrationError,
    plan_symbol_migration,
    transfer_symbol_references,
)


def test_migration_does_not_infer_ownership_from_legacy_file():
    old = fact_node_id("repo", "symbol", "save")
    nodes = [(old, {"type": "CodeSymbol", "repo": "repo", "name": "save", "file_path": "a.py"})]
    points = [
        CodeSymbol(
            id=fact_node_id("repo", "symbol", "save", file),
            name="save",
            kind="symbol",
            repo="repo",
            file_path=file,
        )
        for file in ("a.py", "b.py")
    ]
    edges = [(uuid4(), old, "describes", {"evidence": "manual link"})]
    with pytest.raises(SymbolMigrationError) as caught:
        plan_symbol_migration(nodes, edges, points)
    assert len(next(iter(caught.value.issues[0]["candidates"].values()))) == 2
    assert edges[0][3] == {"evidence": "manual link"}


@pytest.mark.asyncio
async def test_migration_preserves_every_original_source_run():
    from cognee.infrastructure.databases.provenance import (
        GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
        GRAPH_DELETE_MODE_KEY,
        GRAPH_PROVENANCE_VERSION,
        GRAPH_PROVENANCE_VERSION_KEY,
    )
    from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
    from cognee.infrastructure.databases.provenance.source_refs import (
        make_source_ref_key,
        make_source_run_ref,
    )

    old, source = fact_node_id("repo", "symbol", "save"), uuid4()
    new = CodeSymbol(
        id=fact_node_id("repo", "symbol", "save", "a.py"),
        name="save",
        kind="symbol",
        repo="repo",
        file_path="a.py",
    )
    nodes = [(old, {"type": "CodeSymbol", "repo": "repo", "name": "save"})]
    edges = [(source, old, "describes", {"evidence": "original"})]
    migration = plan_symbol_migration(nodes, edges, [new])
    ref, runs = make_source_ref_key(uuid4(), uuid4()), [uuid4(), uuid4()]
    engine = AsyncMock()
    engine.get_graph_metadata.return_value = {
        GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
        GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
    }
    engine.get_edge_delete_data.return_value = {
        EdgeIdentity(str(source), str(old), "describes"): SimpleNamespace(
            source_ref_keys=[ref], source_run_refs=[make_source_run_ref(run, ref) for run in runs]
        )
    }
    await transfer_symbol_references(engine, migration, edges)
    assert engine.add_edges.await_args.args[0][0][3]["evidence"] == "original"
    assert {
        call.kwargs["pipeline_run_id"] for call in engine.attach_edge_source_refs.await_args_list
    } == set(map(str, runs))


@pytest.mark.parametrize(
    "language,source,target,file",
    [
        ("python", "api/routes.py", "orders.service", "orders/service.py"),
        ("python", "api/routes.py", ".helpers", "api/helpers.py"),
        ("typescript", "src/Card.tsx", "../hooks/useOrder", "hooks/useOrder.ts"),
        ("javascript", "src/app.js", "../hooks", "hooks/index.js"),
    ],
)
def test_import_resolution_uses_known_files(language, source, target, file):
    import posixpath

    origin, destination = posixpath.dirname(source), posixpath.dirname(file)
    facts = [
        {"kind": "module", "repo": "r", "name": module} for module in {origin, destination}
    ] + [
        {"kind": "symbol", "repo": "r", "name": "handler", "file": file},
        {
            "kind": "dependency",
            "repo": "r",
            "name": "import",
            "file": source,
            "props": {"source": "internal", "language": language},
            "relations": [{"kind": "imports", "target": target}],
        },
    ]
    edges, skipped = build_code_graph_edges(facts)
    assert skipped == 0
    assert (
        fact_node_id("r", "dependency", "import"),
        fact_node_id("r", "module", destination),
    ) in {(s, t) for s, t, _, _ in edges}


def test_external_alias_missing_and_ambiguous_imports_never_guess():
    facts = [
        {"kind": "module", "repo": "r", "name": "lib"},
        {"kind": "module", "repo": "other", "name": "missing"},
        {"kind": "symbol", "repo": "r", "name": "lib.a", "file": "lib/index.ts"},
        {"kind": "symbol", "repo": "r", "name": "lib.b", "file": "lib/index.js"},
    ]
    for name, source in (
        ("lib", "internal"),
        ("@lib", "internal"),
        ("missing", "internal"),
        ("external", "external"),
    ):
        facts.append(
            {
                "kind": "dependency",
                "repo": "r",
                "name": name,
                "file": "src/a.ts",
                "props": {"source": source, "language": "typescript"},
                "relations": [
                    {"kind": "imports", "target": name + "/index" if name == "lib" else name}
                ],
            }
        )
    diagnostics = []
    edges, skipped = build_code_graph_edges(facts, diagnostics=diagnostics)
    assert edges == [] and skipped == 4
    assert len(diagnostics) == 4


def test_coverage_unknown_partial_and_no_observed_gaps_are_distinct_and_copied():
    quality = {
        "files_seen": 1,
        "files_parsed": 1,
        "files_skipped": 0,
        "dirs_skipped": 0,
        "parse_errors": 0,
    }
    repo = {
        "name": "r",
        "last_receipt": {"quality": quality},
        "last_delta": {"mapping": {"relations_skipped": 0, "by_reason": {}}},
    }

    def report():
        return repository_coverage({"id": repo})["repositories"][0]

    assert report()["status"] == "observed_no_gaps"
    report()["extraction"]["files_parsed"] = 999
    assert quality["files_parsed"] == 1
    repo["last_delta"]["mapping"] = {
        "relations_skipped": 3,
        "by_reason": {"external_dependency": 3},
    }
    assert report()["status"] == "observed_no_gaps"
    quality["files_skipped"] = 4  # Files the extractor ignores are not gaps.
    assert report()["status"] == "observed_no_gaps"
    repo["last_delta"]["mapping"]["by_reason"]["ambiguous"] = 1
    assert report()["status"] == "partial"
    repo.pop("last_delta")
    assert report()["status"] == "unknown"
    repo["last_receipt"] = {"quality": {}}
    assert report()["status"] == "unknown"


def test_all_generated_relationships_have_loader_ownership():
    from cognee.tasks.code_graph.symbol_migration import CODE_EDGE_OWNER

    facts = [
        {"kind": "module", "repo": "r", "name": "src"},
        {"kind": "symbol", "repo": "r", "name": "src.helper", "file": "src/a.py"},
        {
            "kind": "symbol",
            "repo": "r",
            "name": "src.main",
            "file": "src/a.py",
            "relations": [
                {"kind": "calls", "target": "src.helper"},
                {"kind": "declares", "target": "src"},
            ],
        },
    ]
    edges, skipped = build_code_graph_edges(facts)
    assert skipped == 0 and {r for _, _, r, _ in edges} == {"calls", "declares"}
    assert all(props["producer"] == CODE_EDGE_OWNER for _, _, _, props in edges)


def test_name_only_call_prefers_unique_same_file_symbol():
    facts = [
        {"kind": "symbol", "repo": "r", "name": "save", "file": file} for file in ("a.py", "b.py")
    ] + [
        {
            "kind": "symbol",
            "repo": "r",
            "name": "main",
            "file": "a.py",
            "relations": [{"kind": "calls", "target": "save"}],
        }
    ]
    edges, skipped = build_code_graph_edges(facts)
    assert skipped == 0
    assert edges[0][1] == fact_node_id("r", "symbol", "save", "a.py")


def test_import_lookup_does_not_scan_repository_files_per_query():
    from cognee.tasks.code_graph.imports import ImportIndex

    facts = [{"kind": "module", "repo": "r", "name": "pkg"}] + [
        {"kind": "symbol", "repo": "r", "name": str(i), "file": f"pkg/f{i}.py"}
        for i in range(20000)
    ]
    index = ImportIndex(facts, "r")
    index.files = None  # Queries must use the prebuilt lookup, not the inventory.
    for i in range(500):
        assert (
            index.resolve(
                {"file": "main.py", "props": {"source": "internal", "language": "python"}},
                f"pkg.f{i}",
                "r",
            )["name"]
            == "pkg"
        )


@pytest.mark.parametrize(
    "language,target,file,module,props",
    [
        (
            "go",
            "github.com/acme/r/internal/auth",
            "internal/auth/auth.go",
            "internal/auth",
            {"modulePath": "github.com/acme/r"},
        ),
        ("java", "com.acme.Auth", "com/acme/Auth.java", "com/acme", {}),
        ("rust", "crate::auth", "src/auth.rs", "src", {}),
    ],
)
def test_additional_language_imports(language, target, file, module, props):
    from cognee.tasks.code_graph.imports import ImportIndex

    index = ImportIndex(
        [
            {"kind": "module", "repo": "r", "name": ".", "props": props},
            {"kind": "module", "repo": "r", "name": module},
            {"kind": "symbol", "repo": "r", "name": "Auth", "file": file},
        ],
        "r",
    )
    assert (
        index.resolve(
            {"file": "main", "props": {"source": "internal", "language": language}}, target, "r"
        )["name"]
        == module
    )


def test_external_import_links_only_to_declared_package():
    facts = [
        {"kind": "module", "repo": "r", "name": "react"},
        {
            "kind": "dependency",
            "repo": "r",
            "name": "pkg:npm/react",
            "props": {"type": "package", "package_name": "react"},
        },
        {
            "kind": "dependency",
            "repo": "r",
            "name": "import-react",
            "file": "src/a.ts",
            "props": {"source": "external", "language": "typescript"},
            "relations": [{"kind": "imports", "target": "react"}],
        },
    ]
    edges, skipped = build_code_graph_edges(facts)
    assert skipped == 0 and len(edges) == 1
    assert edges[0][1] == fact_node_id("r", "dependency", "pkg:npm/react")


def test_code_tool_enums_match_retriever_operations():
    from cognee.api.v1.responses.default_tools import DEFAULT_TOOLS
    from cognee.api.v1.responses.routers.default_tools import DEFAULT_TOOLS as ROUTER_TOOLS
    from cognee.cli.code_search import CODE_OPERATIONS
    from cognee.modules.retrieval.code_retriever import _OPERATIONS

    for tools in (DEFAULT_TOOLS, ROUTER_TOOLS):
        search = next(tool for tool in tools if tool["name"] == "search")
        assert (
            set(search["parameters"]["properties"]["code_query"]["properties"]["operation"]["enum"])
            == _OPERATIONS
        )
    assert set(CODE_OPERATIONS) == _OPERATIONS


def test_migration_treats_the_legacy_mappers_code_to_code_edges_as_regenerated():
    """A legacy graph's own declares/calls edges are not references and must not block the rebuild."""
    old = fact_node_id("repo", "symbol", "save")
    caller = fact_node_id("repo", "symbol", "run")
    module = fact_node_id("repo", "module", "src")
    nodes = [
        (old, {"type": "CodeSymbol", "repo": "repo", "name": "save"}),
        (caller, {"type": "CodeSymbol", "repo": "repo", "name": "run"}),
        (module, {"type": "CodeModule", "repo": "repo", "name": "src"}),
    ]
    points = [
        CodeSymbol(
            id=fact_node_id("repo", "symbol", "save", file),
            name="save",
            kind="symbol",
            repo="repo",
            file_path=file,
        )
        for file in ("a.ts", "b.ts")
    ]
    edges = [
        (caller, old, "calls", {}),
        (module, old, "declares", {}),
        (uuid4(), old, "describes", {"evidence": "manual link"}),
    ]
    with pytest.raises(SymbolMigrationError) as caught:
        plan_symbol_migration(nodes, edges, points)
    assert [issue["relationship"] for issue in caught.value.issues] == ["describes"]


def test_external_dependencies_are_counted_but_not_listed_as_unresolved():
    facts = [
        {
            "kind": "dependency",
            "repo": "r",
            "name": "react",
            "file": "src/a.ts",
            "props": {"source": "external", "language": "typescript"},
            "relations": [{"kind": "imports", "target": "react"}],
        }
    ]
    diagnostics = []
    build_code_graph_edges(facts, diagnostics=diagnostics)
    assert [item["reason"] for item in diagnostics] == ["external_dependency"]


def test_esm_js_import_resolves_the_typescript_file():
    facts = [
        {"kind": "module", "repo": "r", "name": "src"},
        {"kind": "symbol", "repo": "r", "name": "foo.run", "file": "src/foo.ts"},
        {
            "kind": "dependency",
            "repo": "r",
            "name": "import",
            "file": "src/a.ts",
            "props": {"source": "internal", "language": "typescript"},
            "relations": [{"kind": "imports", "target": "./foo.js"}],
        },
    ]
    edges, skipped = build_code_graph_edges(facts)
    assert skipped == 0
    assert any(relation == "imports" for _s, _t, relation, _p in edges)


def test_import_index_only_resolves_internal_imports():
    from cognee.tasks.code_graph.imports import ImportIndex

    facts = [
        {"kind": "module", "repo": "r", "name": "react"},
        {"kind": "symbol", "repo": "r", "name": "react.render", "file": "react/index.js"},
    ]
    index = ImportIndex(facts, "r")
    dependency = {
        "kind": "dependency",
        "file": "src/a.ts",
        "props": {"source": "external", "language": "typescript"},
    }
    assert index.resolve(dependency, "react", "r") is None
    dependency["props"]["source"] = "internal"
    assert index.resolve(dependency, "react", "r")["name"] == "react"


def test_client_route_caller_with_several_same_named_symbols_is_not_guessed():
    facts = [
        {"kind": "symbol", "repo": "r", "name": "handler", "file": "a.ts"},
        {"kind": "symbol", "repo": "r", "name": "handler", "file": "b.ts"},
        {
            "kind": "route",
            "repo": "r",
            "name": "GET /x",
            "file": "c.ts",
            "props": {"role": "client", "caller": "handler"},
        },
    ]
    diagnostics = []
    edges, _ = build_code_graph_edges(facts, diagnostics=diagnostics)
    assert not any(relation == "makes_request" for _s, _t, relation, _p in edges)
    assert [item["reason"] for item in diagnostics] == ["unresolved_caller"]


def test_migration_skips_edges_the_snapshot_owns():
    old = fact_node_id("repo", "symbol", "save")
    nodes = [(old, {"type": "CodeSymbol", "repo": "repo", "name": "save"})]
    points = [
        CodeSymbol(
            id=fact_node_id("repo", "symbol", "save", file),
            name="save",
            kind="symbol",
            repo="repo",
            file_path=file,
        )
        for file in ("a.py", "b.py")
    ]
    owned = (uuid4(), old, "describes", {"producer": "cognee_code_graph"})
    assert plan_symbol_migration(nodes, [owned], points).edges == []


def test_esm_js_rule_applies_to_javascript_family_languages_only():
    from cognee.tasks.code_graph.imports import ImportIndex

    facts = [
        {"kind": "module", "repo": "r", "name": "src"},
        {"kind": "symbol", "repo": "r", "name": "foo.run", "file": "src/foo.ts"},
    ]
    index = ImportIndex(facts, "r")
    dependency = {"kind": "dependency", "file": "src/a.go", "props": {"source": "internal"}}
    for language, resolves in (("typescript", True), ("javascript", True), ("go", False)):
        dependency["props"]["language"] = language
        assert (index.resolve(dependency, "./foo.js", "r") is not None) is resolves


def test_a_short_name_matching_several_facts_is_ambiguous_not_unresolved():
    facts = [
        {"kind": "symbol", "repo": "r", "name": "pkg.a.Database", "file": "pkg/a.py"},
        {"kind": "symbol", "repo": "r", "name": "pkg.b.Database", "file": "pkg/b.py"},
        {
            "kind": "symbol",
            "repo": "r",
            "name": "run",
            "file": "run.py",
            "relations": [{"kind": "calls", "target": "Database"}],
        },
    ]
    diagnostics = []
    build_code_graph_edges(facts, diagnostics=diagnostics)
    assert [item["reason"] for item in diagnostics] == ["ambiguous"]


def test_go_module_path_prefixes_are_found_longest_first_at_separators():
    from cognee.tasks.code_graph.imports import ImportIndex

    facts = [
        {"kind": "module", "repo": "a", "name": "a", "props": {"modulePath": "ex.com/lib"}},
        {"kind": "module", "repo": "b", "name": "b", "props": {"modulePath": "ex.com/lib/sub"}},
    ]
    index = ImportIndex(facts, "a")
    assert list(index.module_path_prefixes("ex.com/lib/sub.Run", "/.")) == [
        "ex.com/lib/sub",
        "ex.com/lib",
    ]
    assert list(index.module_path_prefixes("ex.com/library.Run", "/.")) == []
