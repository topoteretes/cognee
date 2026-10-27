"""Source-file identity and exact links for same-named declarations."""

from cognee.tasks.code_graph.extract_code_graph import (
    build_code_graph_edges,
    fact_node_id,
    map_facts_to_data_points,
)


def test_same_named_symbols_keep_exact_target_ids_and_source_edges():
    facts = [
        {"kind": "symbol", "name": "src.save", "repo": "r", "file": "src/a.ts", "id": "a" * 32},
        {"kind": "symbol", "name": "src.save", "repo": "r", "file": "src/b.ts", "id": "b" * 32},
        {
            "kind": "symbol",
            "name": "src.run",
            "repo": "r",
            "file": "src/a.ts",
            "relations": [{"kind": "calls", "target": "src.save", "target_id": "a" * 32}],
        },
        {
            "kind": "symbol",
            "name": "src.run",
            "repo": "r",
            "file": "src/b.ts",
            "relations": [{"kind": "calls", "target": "src.save", "target_id": "b" * 32}],
        },
        {
            "kind": "symbol",
            "name": "unresolved",
            "repo": "r",
            "file": "src/c.ts",
            "relations": [{"kind": "calls", "target": "src.save"}],
        },
    ]
    points = map_facts_to_data_points(facts)
    assert len(points) == 6  # Five declarations plus their repository.
    edges, skipped = build_code_graph_edges(facts)
    assert skipped == 1
    assert {(source, target, kind) for source, target, kind, _ in edges} == {
        (
            fact_node_id("r", "symbol", "src.run", file),
            fact_node_id("r", "symbol", "src.save", file),
            "calls",
        )
        for file in ("src/a.ts", "src/b.ts")
    }
    assert {point.id for point in points} == {
        point.id for point in map_facts_to_data_points(list(reversed(facts)))
    }
    shifted = [{**fact, "line": 100} for fact in facts]
    assert {point.id for point in points} == {
        point.id for point in map_facts_to_data_points(shifted)
    }


def test_type_membership_stays_in_its_source_file():
    facts = [
        {"kind": "symbol", "repo": "r", "name": name, "file": file, "props": {"symbol_kind": kind}}
        for file in ("a.ts", "b.ts")
        for name, kind in (("src.Store", "class"), ("src.Store.save", "method"))
    ]
    edges, skipped = build_code_graph_edges(facts)
    assert skipped == 0
    assert {(a, b, kind) for a, b, kind, _ in edges} == {
        (
            fact_node_id("r", "symbol", "src.Store", file),
            fact_node_id("r", "symbol", "src.Store.save", file),
            "has_method",
        )
        for file in ("a.ts", "b.ts")
    }


def test_declaration_view_does_not_hide_a_second_symbol_in_the_same_repo():
    facts = [
        {"kind": "symbol", "repo": "r", "name": "Store", "file": "a.py"},
        {"kind": "symbol", "repo": "r", "name": "Store", "file": "b.py"},
        {"kind": "storage", "repo": "r", "name": "Store", "file": "a.py"},
        {
            "kind": "symbol",
            "repo": "r",
            "name": "caller",
            "file": "c.py",
            "relations": [{"kind": "instantiates", "target": "Store"}],
        },
    ]
    assert build_code_graph_edges(facts) == ([], 1)
