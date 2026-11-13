"""Relations into a name a framework extractor gave both a symbol and a view of it.

Since enola 0.4.29 the PHP extractor writes, for every Eloquent model, a
``storage`` fact named exactly like the model class and, for every relation
method, an ``association`` fact named exactly like the method -- same repo,
same file. The JPA and Rails extractors do the same for entities and models.
The writer then leaves ``target_id`` off every relation into those names (two
identities share them), so build_code_graph_edges resolves them by name. The
facts below are shaped like enola 0.4.30's output for a Laravel app.
"""

from cognee.tasks.code_graph.extract_code_graph import (
    build_code_graph_edges,
    fact_node_id,
    map_facts_to_data_points,
)

REPO = "lv"
MODEL = "App\\Models\\User"
MODEL_FILE = "app/Models/User.php"
RELATION = "App\\Models\\User::posts"
SHOW = "App\\Http\\Controllers\\UserController::show"
STORAGE_ENOLA_ID = "4f6ccd8133ea81d75d0e70d041ae25e4"


def _model_facts(repo=REPO, *, storage_file=MODEL_FILE, symbol_file=MODEL_FILE):
    """An Eloquent model: the class and its storage, a relation method and its association."""
    return [
        {
            "kind": "symbol",
            "name": MODEL,
            "file": symbol_file,
            "line": 8,
            "repo": repo,
            "props": {"symbol_kind": "class", "eloquent_table": "users"},
        },
        {
            "kind": "storage",
            "name": MODEL,
            "file": storage_file,
            "line": 8,
            "repo": repo,
            "props": {"framework": "eloquent", "storage_kind": "model", "table": "users"},
            "id": STORAGE_ENOLA_ID,
        },
        {
            "kind": "symbol",
            "name": RELATION,
            "file": MODEL_FILE,
            "line": 14,
            "repo": repo,
            "props": {"symbol_kind": "method", "eloquent_relation": "hasMany"},
        },
        {
            "kind": "association",
            "name": RELATION,
            "file": MODEL_FILE,
            "line": 14,
            "repo": repo,
            "props": {"framework": "eloquent", "model": MODEL, "target": "App\\Models\\Post"},
        },
    ]


def _caller(*relations, name=SHOW, repo=REPO):
    """A controller method whose relations carry no target_id, as enola writes them."""
    return {
        "kind": "symbol",
        "name": name,
        "file": "app/Http/Controllers/UserController.php",
        "repo": repo,
        "props": {"symbol_kind": "method"},
        "relations": [{"kind": kind, "target": target} for kind, target in relations],
    }


def _edges(edges):
    """{(source id, relation, target id)} of built edge tuples."""
    return {(source, relation, target) for source, target, relation, _properties in edges}


def test_a_model_name_resolves_to_its_class_not_the_storage_view():
    caller = _caller(("instantiates", MODEL), ("names", MODEL))
    # A writer-resolved target_id still wins: enola named the storage itself.
    caller["relations"].append(
        {"kind": "depends_on", "target": MODEL, "target_id": STORAGE_ENOLA_ID}
    )
    facts = [*_model_facts(), caller]

    edges, skipped = build_code_graph_edges(facts)

    show = fact_node_id(REPO, "symbol", SHOW)
    model = fact_node_id(REPO, "symbol", MODEL)
    storage = fact_node_id(REPO, "storage", MODEL)
    assert skipped == 0
    assert _edges(edges) == {
        (show, "instantiates", model),
        (show, "names", model),
        (show, "depends_on", storage),
    }
    # Both ends are nodes the mapping stores.
    node_ids = {data_point.id for data_point in map_facts_to_data_points(facts)}
    assert {model, storage, show} <= node_ids


def test_a_relation_method_name_resolves_to_the_method_not_the_association_view():
    facts = [*_model_facts(), _caller(("calls", RELATION))]

    edges, skipped = build_code_graph_edges(facts)

    assert skipped == 0
    assert _edges(edges) == {
        (
            fact_node_id(REPO, "symbol", SHOW),
            "calls",
            fact_node_id(REPO, "symbol", RELATION),
        )
    }


def test_a_same_named_fact_from_another_file_stays_a_rival():
    moved = _model_facts(storage_file="database/migrations/2024_01_01_create_users_table.php")
    edges, skipped = build_code_graph_edges([*moved, _caller(("instantiates", MODEL))])
    assert (edges, skipped) == ([], 1)

    # Without a file on the symbol nothing ties the two together either.
    unplaced = _model_facts(storage_file=None, symbol_file=None)
    edges, skipped = build_code_graph_edges([*unplaced, _caller(("instantiates", MODEL))])
    assert (edges, skipped) == ([], 1)

    # Nor does a view that has no file of its own.
    unfiled_view = _model_facts(storage_file=None)
    edges, skipped = build_code_graph_edges([*unfiled_view, _caller(("instantiates", MODEL))])
    assert (edges, skipped) == ([], 1)


def test_two_same_named_symbols_stay_ambiguous_with_a_view_beside_them():
    facts = [
        *_model_facts("a"),
        {"kind": "symbol", "name": MODEL, "file": MODEL_FILE, "repo": "b"},
        _caller(("instantiates", MODEL), repo="c"),
        _caller(("instantiates", MODEL), name="App\\Jobs\\Sync::handle", repo="a"),
    ]

    edges, skipped = build_code_graph_edges(facts)

    # Repo c has no candidate of its own and two symbols to choose from: skipped.
    # Repo a still prefers its own symbol, its storage view no longer in the way.
    assert skipped == 1
    assert _edges(edges) == {
        (
            fact_node_id("a", "symbol", "App\\Jobs\\Sync::handle"),
            "instantiates",
            fact_node_id("a", "symbol", MODEL),
        )
    }


def test_a_view_in_another_repo_is_not_merged_into_the_symbol():
    symbol, storage = _model_facts("a")[:2]
    storage = {**storage, "repo": "b"}
    facts = [
        symbol,
        storage,
        _caller(("instantiates", MODEL), repo="c"),
        _caller(("instantiates", MODEL), name="App\\Jobs\\Sync::handle", repo="b"),
    ]

    edges, skipped = build_code_graph_edges(facts)

    # From repo c the two are rivals; repo b keeps resolving to its own fact.
    assert skipped == 1
    assert _edges(edges) == {
        (
            fact_node_id("b", "symbol", "App\\Jobs\\Sync::handle"),
            "instantiates",
            fact_node_id("b", "storage", MODEL),
        )
    }


def test_a_jpa_entity_resolves_by_qualified_and_bare_name():
    entity = "src/main/java/org/acme/owner.Owner"
    entity_file = "src/main/java/org/acme/owner/Owner.java"
    facts = [
        {
            "kind": "symbol",
            "name": entity,
            "file": entity_file,
            "repo": "petclinic",
            "props": {"symbol_kind": "class"},
        },
        {
            "kind": "storage",
            "name": entity,
            "file": entity_file,
            "repo": "petclinic",
            "props": {"framework": "jpa", "storage_kind": "entity", "table": "owners"},
        },
        _caller(
            ("instantiates", entity),
            ("calls", "Owner"),
            name="src/main/java/org/acme/owner.OwnerController.findOwner",
            repo="petclinic",
        ),
    ]

    edges, skipped = build_code_graph_edges(facts)

    finder = fact_node_id(
        "petclinic", "symbol", "src/main/java/org/acme/owner.OwnerController.findOwner"
    )
    owner = fact_node_id("petclinic", "symbol", entity)
    assert skipped == 0
    assert _edges(edges) == {(finder, "instantiates", owner), (finder, "calls", owner)}


def test_a_bare_name_with_a_rival_symbol_elsewhere_stays_skipped():
    entity = "src/main/java/org/acme/owner.Owner"
    entity_file = "src/main/java/org/acme/owner/Owner.java"
    finder_name = "src/main/java/org/acme/owner.OwnerController.findOwner"
    facts = [
        {"kind": "symbol", "name": entity, "file": entity_file, "repo": "petclinic"},
        {"kind": "storage", "name": entity, "file": entity_file, "repo": "petclinic"},
        # A second class with the same bare name in another package.
        {
            "kind": "symbol",
            "name": "src/main/java/org/acme/legacy.Owner",
            "file": "src/main/java/org/acme/legacy/Owner.java",
            "repo": "petclinic",
        },
        _caller(("instantiates", entity), ("calls", "Owner"), name=finder_name, repo="petclinic"),
    ]

    edges, skipped = build_code_graph_edges(facts)

    # The qualified name still means the entity class; the bare name has two
    # classes to choose from once the storage view steps aside, so it is skipped.
    assert skipped == 1
    assert _edges(edges) == {
        (
            fact_node_id("petclinic", "symbol", finder_name),
            "instantiates",
            fact_node_id("petclinic", "symbol", entity),
        )
    }


def test_insight_evidence_cites_the_model_class():
    insight = {
        "kind": "insight",
        "name": "God class: App\\Models\\User",
        "props": {"source": "god-class", "confidence": 0.8},
        "relations": [{"kind": "evidences", "target": MODEL}],
    }

    edges, skipped = build_code_graph_edges([*_model_facts(), insight])

    assert skipped == 0
    assert _edges(edges) == {
        (
            fact_node_id(REPO, "insight", insight["name"]),
            "evidences",
            fact_node_id(REPO, "symbol", MODEL),
        )
    }
