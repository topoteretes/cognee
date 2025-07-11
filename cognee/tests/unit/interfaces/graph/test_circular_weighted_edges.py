import pytest
from typing import List, Any
from pydantic import SkipValidation
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.graph.utils import get_graph_from_model


class CircularUser(DataPoint):
    name: str
    email: str
    friends: SkipValidation[Any] = None  # (Edge, list["CircularUser"])
    best_friend: SkipValidation[Any] = None  # (Edge, CircularUser)
    metadata: dict = {"index_fields": ["name", "email"]}


class CircularCompany(DataPoint):
    name: str
    subsidiaries: SkipValidation[Any] = None  # (Edge, list["CircularCompany"])
    parent_company: SkipValidation[Any] = None  # (Edge, "CircularCompany")
    employees: SkipValidation[Any] = None  # (Edge, list[CircularUser])
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_circular_reference_with_weights():
    """Test circular references with weighted edges don't cause infinite loops"""

    # Create users with circular friendship
    user1 = CircularUser(name="Alice", email="alice@example.com")
    user2 = CircularUser(name="Bob", email="bob@example.com")
    user3 = CircularUser(name="Carol", email="carol@example.com")

    # Create circular friendships with weights
    user1.friends = (
        Edge(
            weights={"friendship_strength": 0.9, "trust_level": 0.8, "years_known": 0.7},
            relationship_type="friends_with",
        ),
        [user2, user3],
    )

    user2.friends = (
        Edge(
            weights={"friendship_strength": 0.8, "trust_level": 0.9, "years_known": 0.5},
            relationship_type="friends_with",
        ),
        [user1, user3],
    )

    user3.friends = (
        Edge(
            weights={"friendship_strength": 0.7, "trust_level": 0.8, "years_known": 0.6},
            relationship_type="friends_with",
        ),
        [user1, user2],
    )

    # Add best friend relationships (single circular reference)
    user1.best_friend = (
        Edge(weight=0.95, weights={"emotional_bond": 0.9}, relationship_type="best_friends_with"),
        user2,
    )

    user2.best_friend = (
        Edge(weight=0.95, weights={"emotional_bond": 0.9}, relationship_type="best_friends_with"),
        user1,
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    # This should not cause infinite loop
    nodes, edges = await get_graph_from_model(user1, added_nodes, added_edges, visited_properties)

    # Should have all 3 users
    assert len(nodes) == 3, f"Expected 3 nodes, got {len(nodes)}"
    # Should have edges but not infinite
    assert len(edges) > 0, "Should have edges"
    assert len(edges) < 20, f"Too many edges, possible infinite loop: {len(edges)}"

    # Verify visited_properties is populated
    assert len(visited_properties) > 0, "visited_properties should track visited relationships"

    # Verify weights are preserved in circular relationships
    friendship_edges = [e for e in edges if e[2] == "friends_with"]
    best_friend_edges = [e for e in edges if e[2] == "best_friends_with"]

    assert len(friendship_edges) > 0, "Should have friendship edges"
    assert len(best_friend_edges) > 0, "Should have best friend edges"

    # Check that weights are properly stored
    for edge in friendship_edges:
        props = edge[3]
        assert "weight_friendship_strength" in props
        assert "weight_trust_level" in props
        assert "weight_years_known" in props

    for edge in best_friend_edges:
        props = edge[3]
        assert "weight" in props
        assert props["weight"] == 0.95
        assert "weight_emotional_bond" in props
        assert props["weight_emotional_bond"] == 0.9


@pytest.mark.asyncio
async def test_deep_circular_hierarchy_with_weights():
    """Test deep circular hierarchy with weighted relationships"""

    # Create companies with circular ownership
    parent_corp = CircularCompany(name="Parent Corp")
    subsidiary1 = CircularCompany(name="Subsidiary 1")
    subsidiary2 = CircularCompany(name="Subsidiary 2")

    # Create employees
    ceo = CircularUser(name="CEO", email="ceo@parent.com")
    manager1 = CircularUser(name="Manager 1", email="m1@sub1.com")
    manager2 = CircularUser(name="Manager 2", email="m2@sub2.com")

    # Set up circular company relationships
    parent_corp.subsidiaries = (
        Edge(
            weights={
                "ownership_percentage": 0.8,
                "control_level": 0.9,
                "strategic_importance": 0.7,
            },
            relationship_type="owns",
        ),
        [subsidiary1, subsidiary2],
    )

    subsidiary1.parent_company = (
        Edge(
            weights={"dependence_level": 0.8, "revenue_contribution": 0.6},
            relationship_type="owned_by",
        ),
        parent_corp,
    )

    subsidiary2.parent_company = (
        Edge(
            weights={"dependence_level": 0.7, "revenue_contribution": 0.4},
            relationship_type="owned_by",
        ),
        parent_corp,
    )

    # Add employee relationships with circular references
    parent_corp.employees = (
        Edge(
            weights={"seniority": 0.95, "performance": 0.9, "leadership": 0.95},
            relationship_type="employs",
        ),
        [ceo],
    )

    subsidiary1.employees = (
        Edge(
            weights={"seniority": 0.7, "performance": 0.8, "leadership": 0.6},
            relationship_type="employs",
        ),
        [manager1],
    )

    subsidiary2.employees = (
        Edge(
            weights={"seniority": 0.6, "performance": 0.7, "leadership": 0.5},
            relationship_type="employs",
        ),
        [manager2],
    )

    # Create cross-company friendships (more circular references)
    ceo.friends = (
        Edge(
            weights={"professional_relationship": 0.8, "personal_friendship": 0.6},
            relationship_type="colleagues_with",
        ),
        [manager1, manager2],
    )

    manager1.friends = (
        Edge(
            weights={"professional_relationship": 0.7, "personal_friendship": 0.8},
            relationship_type="colleagues_with",
        ),
        [ceo, manager2],
    )

    manager2.friends = (
        Edge(
            weights={"professional_relationship": 0.6, "personal_friendship": 0.7},
            relationship_type="colleagues_with",
        ),
        [ceo, manager1],
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    # Start from parent company - should traverse entire network without infinite loop
    nodes, edges = await get_graph_from_model(
        parent_corp, added_nodes, added_edges, visited_properties
    )

    # Should have all entities: 3 companies + 3 users = 6 nodes
    assert len(nodes) == 6, f"Expected 6 nodes, got {len(nodes)}"

    # Should have many edges but not infinite
    assert len(edges) > 0, "Should have edges"
    assert len(edges) < 50, f"Too many edges, possible infinite loop: {len(edges)}"

    # Verify all expected relationship types exist
    relationship_types = {edge[2] for edge in edges}
    expected_types = {"owns", "owned_by", "employs", "colleagues_with"}

    for expected_type in expected_types:
        assert expected_type in relationship_types, f"Missing relationship type: {expected_type}"

    # Verify visited_properties prevents circular traversal
    assert len(visited_properties) > 0, "visited_properties should track visited relationships"

    # Check that multiple visits to same property are prevented
    property_keys = list(visited_properties.keys())
    unique_keys = set(property_keys)
    assert len(property_keys) == len(unique_keys), "Should not have duplicate property keys"


@pytest.mark.asyncio
async def test_self_referencing_with_weights():
    """Test self-referencing nodes with weights"""

    # Create a user who is their own friend (edge case)
    narcissist = CircularUser(name="Narcissist", email="me@myself.com")

    narcissist.best_friend = (
        Edge(weight=1.0, weights={"self_love": 1.0, "confidence": 0.95}, relationship_type="loves"),
        narcissist,  # Self-reference
    )

    narcissist.friends = (
        Edge(
            weights={"self_appreciation": 1.0, "loneliness": 0.8}, relationship_type="friends_with"
        ),
        [narcissist],  # Self-reference in list
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    # Should handle self-references without infinite loop
    nodes, edges = await get_graph_from_model(
        narcissist, added_nodes, added_edges, visited_properties
    )

    # Should have only 1 node (the narcissist)
    assert len(nodes) == 1, f"Expected 1 node, got {len(nodes)}"

    # Should have 2 self-referencing edges
    assert len(edges) == 2, f"Expected 2 edges, got {len(edges)}"

    # Both edges should point to the same node
    for edge in edges:
        source_id, target_id, relationship_name, edge_properties = edge
        assert source_id == target_id, "Self-referencing edge should have same source and target"

        # Check weights are preserved
        if relationship_name == "loves":
            assert "weight" in edge_properties
            assert edge_properties["weight"] == 1.0
            assert "weight_self_love" in edge_properties
            assert "weight_confidence" in edge_properties
        elif relationship_name == "friends_with":
            assert "weight_self_appreciation" in edge_properties
            assert "weight_loneliness" in edge_properties


@pytest.mark.asyncio
async def test_visited_properties_tracking():
    """Test that visited_properties correctly tracks and prevents revisiting"""

    user1 = CircularUser(name="User1", email="u1@test.com")
    user2 = CircularUser(name="User2", email="u2@test.com")

    # Create mutual friendship
    user1.friends = (Edge(weight=0.8, relationship_type="friends_with"), [user2])

    user2.friends = (Edge(weight=0.8, relationship_type="friends_with"), [user1])

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    await get_graph_from_model(user1, added_nodes, added_edges, visited_properties)

    # Check that visited_properties contains the expected keys
    expected_keys = [f"{user1.id}friends_with{user2.id}", f"{user2.id}friends_with{user1.id}"]

    for key in expected_keys:
        assert key in visited_properties, f"Expected key {key} in visited_properties"
        assert visited_properties[key] is True, f"Expected {key} to be marked as visited"

    # Run again with the same visited_properties - should not add duplicate processing
    original_visited_count = len(visited_properties)

    await get_graph_from_model(user1, added_nodes, added_edges, visited_properties)

    # Should not have added new visited properties (already processed)
    assert len(visited_properties) == original_visited_count, (
        "Should not revisit already processed properties"
    )
