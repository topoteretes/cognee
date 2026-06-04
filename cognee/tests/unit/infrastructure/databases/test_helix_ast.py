"""Unit tests for the HelixDB JSON-AST builders.

These assert the exact wire encodings the Helix v2 gateway expects, with no
running server, so they run in CI without HelixDB.
"""

from uuid import UUID

from cognee.infrastructure.databases.graph.helix_driver import ast


def test_property_value_scalars():
    assert ast.property_value(True) == {"Bool": True}
    assert ast.property_value(42) == {"I64": 42}
    assert ast.property_value(3.5) == {"F64": 3.5}
    assert ast.property_value("x") == {"String": "x"}
    uid = UUID("12345678-1234-5678-1234-567812345678")
    assert ast.property_value(uid) == {"String": str(uid)}


def test_property_value_bool_before_int():
    # bool is a subclass of int — must encode as Bool, not I64.
    assert ast.property_value(False) == {"Bool": False}


def test_property_value_complex_is_json_string():
    encoded = ast.property_value({"a": 1})
    assert "String" in encoded
    assert encoded["String"] == '{"a": 1}'


def test_f32_array():
    assert ast.f32_array([0.1, 0.2]) == {"F32Array": [0.1, 0.2]}


def test_eq_predicate():
    assert ast.eq("id", "u-1") == {"Eq": ["id", {"String": "u-1"}]}


def test_and_or_predicates():
    a = ast.eq("type", "Entity")
    b = ast.eq("tenant_id", "default")
    assert ast.and_([a, b]) == {"And": [a, b]}
    assert ast.or_([a, b]) == {"Or": [a, b]}


def test_is_in_predicate():
    assert ast.is_in("id", ["a", "b"]) == {
        "IsIn": ["id", {"Array": [{"String": "a"}, {"String": "b"}]}]
    }


def test_n_where_and_traversal():
    assert ast.n_where(ast.eq("id", "x")) == {"NWhere": {"Eq": ["id", {"String": "x"}]}}
    assert ast.out("FOLLOWS") == {"Out": "FOLLOWS"}
    assert ast.out() == {"Out": None}
    assert ast.both() == {"Both": None}


def test_vector_search_nodes_shapes():
    step = ast.vector_search_nodes("COGNEE_NODE", "Entity_name", [0.1, 0.2], 5)
    vs = step["VectorSearchNodes"]
    assert vs["label"] == "COGNEE_NODE"
    assert vs["property"] == "Entity_name"
    assert vs["query_vector"] == {"Value": {"F32Array": [0.1, 0.2]}}
    assert vs["k"] == {"Literal": 5}
    assert vs["tenant_value"] is None

    partitioned = ast.vector_search_nodes("COGNEE_NODE", "Entity_name", [0.1], 3, tenant_value="t1")
    assert partitioned["VectorSearchNodes"]["tenant_value"] == {"Value": {"String": "t1"}}


def test_add_node_encodes_inputs():
    step = ast.add_node("COGNEE_NODE", {"id": "u-1", "n": 2})
    assert step == {
        "AddN": {
            "label": "COGNEE_NODE",
            "properties": [
                ["id", {"Value": {"String": "u-1"}}],
                ["n", {"Value": {"I64": 2}}],
            ],
        }
    }


def test_add_node_with_inputs_passthrough():
    encoded = [["Entity_name", ast.input_vector([0.1, 0.2])]]
    step = ast.add_node_with_inputs("COGNEE_NODE", encoded)
    assert step["AddN"]["properties"][0][1] == {"Value": {"F32Array": [0.1, 0.2]}}


def test_add_edge_targets_var():
    step = ast.add_edge("KNOWS", "tgt0", {"source_id": "a", "target_id": "b"})
    assert step["AddE"]["label"] == "KNOWS"
    assert step["AddE"]["to"] == {"Var": "tgt0"}
    assert ["source_id", {"Value": {"String": "a"}}] in step["AddE"]["properties"]


def test_set_property_and_inputs():
    assert ast.set_property("name", ast.input_value("Bob")) == {
        "SetProperty": ["name", {"Value": {"String": "Bob"}}]
    }
    assert ast.input_string_array(["a", "b"]) == {"Value": {"StringArray": ["a", "b"]}}


def test_project_and_value_map():
    proj = ast.project([ast.projection("$id", "id"), ast.projection("$distance", "distance")])
    assert proj == {
        "Project": [
            {"source": "$id", "alias": "id"},
            {"source": "$distance", "alias": "distance"},
        ]
    }
    assert ast.value_map(None) == {"ValueMap": None}
    assert ast.value_map(["$id", "name"]) == {"ValueMap": ["$id", "name"]}


def test_index_specs():
    eq_idx = ast.create_node_equality_index("COGNEE_NODE", "id", unique=True)
    assert eq_idx == {
        "CreateIndex": {
            "spec": {"NodeEquality": {"label": "COGNEE_NODE", "property": "id", "unique": True}},
            "if_not_exists": True,
        }
    }
    vec_idx = ast.create_node_vector_index(
        "COGNEE_NODE", "Entity_name", tenant_property="tenant_id"
    )
    assert vec_idx["CreateIndex"]["spec"] == {
        "NodeVector": {
            "label": "COGNEE_NODE",
            "property": "Entity_name",
            "tenant_property": "tenant_id",
        }
    }


def test_query_entry():
    entry = ast.query_entry("q", [ast.COUNT], condition={"VarNotEmpty": "x"})
    assert entry == {"Query": {"name": "q", "steps": ["Count"], "condition": {"VarNotEmpty": "x"}}}


def test_unit_constants():
    assert ast.DEDUP == "Dedup"
    assert ast.COUNT == "Count"
    assert ast.DROP == "Drop"
