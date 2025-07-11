import pytest
from typing import List, Any
from pydantic import SkipValidation
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.graph.utils import get_graph_from_model


class Product(DataPoint):
    name: str
    description: str
    metadata: dict = {"index_fields": ["name"]}


class Category(DataPoint):
    name: str
    description: str
    products: List[Product] = []
    metadata: dict = {"index_fields": ["name"]}


class User(DataPoint):
    name: str
    email: str
    # Weighted relationships
    purchased_products: SkipValidation[Any] = None  # (Edge, list[Product])
    favorite_categories: SkipValidation[Any] = None  # (Edge, list[Category])
    follows: SkipValidation[Any] = None  # (Edge, list["User"])
    metadata: dict = {"index_fields": ["name", "email"]}


class Company(DataPoint):
    name: str
    description: str
    employees: SkipValidation[Any] = None  # (Edge, list[User])
    partners: SkipValidation[Any] = None  # (Edge, list["Company"])
    metadata: dict = {"index_fields": ["name"]}


@pytest.mark.asyncio
async def test_single_weight_edge():
    """Test get_graph_from_model with single weight edges (backward compatible)"""

    product1 = Product(name="Laptop", description="Gaming laptop")
    product2 = Product(name="Mouse", description="Wireless mouse")

    user = User(
        name="John Doe",
        email="john@example.com",
        purchased_products=(Edge(weight=0.8, relationship_type="purchased"), [product1, product2]),
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(user, added_nodes, added_edges, visited_properties)

    # Should have user + 2 products = 3 nodes
    assert len(nodes) == 3, f"Expected 3 nodes, got {len(nodes)}"
    # Should have 2 edges (user -> product1, user -> product2)
    assert len(edges) == 2, f"Expected 2 edges, got {len(edges)}"

    # Check edge properties contain weight
    for edge in edges:
        source_id, target_id, relationship_name, edge_properties = edge
        assert "weight" in edge_properties, "Edge should contain weight property"
        assert edge_properties["weight"] == 0.8, (
            f"Expected weight 0.8, got {edge_properties['weight']}"
        )
        assert edge_properties["relationship_name"] == "purchased"


@pytest.mark.asyncio
async def test_multiple_weights_edge():
    """Test get_graph_from_model with multiple weights on edges"""

    category1 = Category(name="Electronics", description="Electronic products")
    category2 = Category(name="Gaming", description="Gaming products")

    user = User(
        name="Alice Smith",
        email="alice@example.com",
        favorite_categories=(
            Edge(
                weights={
                    "interest_level": 0.9,
                    "time_spent": 0.7,
                    "purchase_frequency": 0.8,
                    "expertise": 0.6,
                },
                relationship_type="interested_in",
            ),
            [category1, category2],
        ),
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(user, added_nodes, added_edges, visited_properties)

    # Should have user + 2 categories = 3 nodes
    assert len(nodes) == 3, f"Expected 3 nodes, got {len(nodes)}"
    # Should have 2 edges
    assert len(edges) == 2, f"Expected 2 edges, got {len(edges)}"

    # Check edge properties contain multiple weights
    for edge in edges:
        source_id, target_id, relationship_name, edge_properties = edge
        assert edge_properties["relationship_name"] == "interested_in"

        # Check individual weight fields
        assert "weight_interest_level" in edge_properties
        assert "weight_time_spent" in edge_properties
        assert "weight_purchase_frequency" in edge_properties
        assert "weight_expertise" in edge_properties

        assert edge_properties["weight_interest_level"] == 0.9
        assert edge_properties["weight_time_spent"] == 0.7
        assert edge_properties["weight_purchase_frequency"] == 0.8
        assert edge_properties["weight_expertise"] == 0.6


@pytest.mark.asyncio
async def test_mixed_single_and_multiple_weights():
    """Test get_graph_from_model with both single weight and multiple weights on same edge"""

    product = Product(name="Smartphone", description="Latest smartphone")

    user = User(
        name="Bob Wilson",
        email="bob@example.com",
        purchased_products=(
            Edge(
                weight=0.7,  # Single weight (backward compatible)
                weights={"satisfaction": 0.9, "value_for_money": 0.6},  # Multiple weights
                relationship_type="owns",
            ),
            [product],
        ),
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(user, added_nodes, added_edges, visited_properties)

    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    assert len(edges) == 1, f"Expected 1 edge, got {len(edges)}"

    edge_properties = edges[0][3]

    # Should have both single weight and multiple weights
    assert "weight" in edge_properties, "Should have backward compatible weight field"
    assert edge_properties["weight"] == 0.7

    assert "weight_satisfaction" in edge_properties
    assert "weight_value_for_money" in edge_properties
    assert edge_properties["weight_satisfaction"] == 0.9
    assert edge_properties["weight_value_for_money"] == 0.6


@pytest.mark.asyncio
async def test_complex_weighted_relationships():
    """Test complex scenario with multiple entities and various weighted relationships"""

    # Create products and categories
    product1 = Product(name="Gaming Chair", description="Ergonomic gaming chair")
    product2 = Product(name="Mechanical Keyboard", description="RGB mechanical keyboard")

    category = Category(name="Gaming Accessories", description="Gaming accessories category")
    category.products = [product1, product2]

    # Create users with different weighted relationships
    user1 = User(
        name="Gamer Pro",
        email="gamerpro@example.com",
        purchased_products=(
            Edge(
                weights={
                    "satisfaction": 0.95,
                    "frequency_of_use": 0.9,
                    "recommendation_likelihood": 0.8,
                },
                relationship_type="purchased",
            ),
            [product1, product2],
        ),
        favorite_categories=(Edge(weight=0.9, relationship_type="follows"), [category]),
    )

    user2 = User(
        name="Casual User",
        email="casual@example.com",
        purchased_products=(Edge(weight=0.6, relationship_type="purchased"), [product1]),
    )

    # Create weighted user relationships
    user1.follows = (
        Edge(
            weights={
                "friendship_level": 0.7,
                "shared_interests": 0.8,
                "communication_frequency": 0.5,
            },
            relationship_type="follows",
        ),
        [user2],
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    # Process user1 (which should process all connected nodes)
    nodes, edges = await get_graph_from_model(user1, added_nodes, added_edges, visited_properties)

    # Should have: user1, user2, 2 products, 1 category = 5 nodes
    assert len(nodes) == 5, f"Expected 5 nodes, got {len(nodes)}"

    # Should have multiple edges with different weight configurations
    assert len(edges) > 0, "Should have edges"

    # Verify that different edge types are created correctly
    edge_types = set()
    weighted_edges = 0
    multi_weighted_edges = 0

    for edge in edges:
        source_id, target_id, relationship_name, edge_properties = edge
        edge_types.add(relationship_name)

        if "weight" in edge_properties:
            weighted_edges += 1

        # Count edges with multiple weights
        multi_weight_fields = [k for k in edge_properties.keys() if k.startswith("weight_")]
        if len(multi_weight_fields) > 1:
            multi_weighted_edges += 1

    assert "purchased" in edge_types
    assert "follows" in edge_types
    assert weighted_edges > 0, "Should have edges with weights"
    assert multi_weighted_edges > 0, "Should have edges with multiple weights"


@pytest.mark.asyncio
async def test_company_hierarchy_with_weights():
    """Test hierarchical company structure with weighted relationships"""

    # Create users
    ceo = User(name="CEO", email="ceo@company.com")
    manager = User(name="Manager", email="manager@company.com")
    developer = User(name="Developer", email="dev@company.com")

    # Create companies with weighted employee relationships
    startup = Company(
        name="Tech Startup",
        description="Innovative tech startup",
        employees=(
            Edge(
                weights={"seniority": 0.9, "performance": 0.8, "leadership": 0.95},
                relationship_type="employs",
            ),
            [ceo, manager, developer],
        ),
    )

    corporation = Company(name="Big Corp", description="Large corporation")

    # Create partnership with weights
    startup.partners = (
        Edge(
            weights={"trust_level": 0.7, "business_value": 0.8, "strategic_importance": 0.6},
            relationship_type="partners_with",
        ),
        [corporation],
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(startup, added_nodes, added_edges, visited_properties)

    # Should have: startup, corporation, 3 users = 5 nodes
    assert len(nodes) == 5, f"Expected 5 nodes, got {len(nodes)}"

    # Verify weighted relationships are properly stored
    partnership_edges = [e for e in edges if e[2] == "partners_with"]
    employee_edges = [e for e in edges if e[2] == "employs"]

    assert len(partnership_edges) == 1, "Should have one partnership edge"
    assert len(employee_edges) == 3, "Should have three employee edges"

    # Check partnership edge weights
    partnership_props = partnership_edges[0][3]
    assert "weight_trust_level" in partnership_props
    assert "weight_business_value" in partnership_props
    assert "weight_strategic_importance" in partnership_props

    # Check employee edge weights
    for edge in employee_edges:
        props = edge[3]
        assert "weight_seniority" in props
        assert "weight_performance" in props
        assert "weight_leadership" in props


@pytest.mark.asyncio
async def test_edge_metadata_preservation():
    """Test that all edge metadata is preserved correctly in weighted edges"""

    product = Product(name="Test Product", description="A test product")

    user = User(
        name="Test User",
        email="test@example.com",
        purchased_products=(
            Edge(weight=0.8, weights={"quality": 0.9, "price": 0.7}, relationship_type="purchased"),
            [product],
        ),
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(user, added_nodes, added_edges, visited_properties)

    assert len(edges) == 1, "Should have exactly one edge"

    edge_properties = edges[0][3]

    # Check all required metadata is present
    assert "source_node_id" in edge_properties
    assert "target_node_id" in edge_properties
    assert "relationship_name" in edge_properties
    assert "updated_at" in edge_properties

    # Check relationship type
    assert edge_properties["relationship_name"] == "purchased"

    # Check weights are properly stored
    assert "weight" in edge_properties
    assert edge_properties["weight"] == 0.8

    assert "weight_quality" in edge_properties
    assert edge_properties["weight_quality"] == 0.9

    assert "weight_price" in edge_properties
    assert edge_properties["weight_price"] == 0.7


@pytest.mark.asyncio
async def test_no_weights_edge():
    """Test that edges without weights still work correctly"""

    product = Product(name="Simple Product", description="No weights product")

    user = User(
        name="Simple User",
        email="simple@example.com",
        purchased_products=(Edge(relationship_type="purchased"), [product]),
    )

    added_nodes = {}
    added_edges = {}
    visited_properties = {}

    nodes, edges = await get_graph_from_model(user, added_nodes, added_edges, visited_properties)

    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    assert len(edges) == 1, f"Expected 1 edge, got {len(edges)}"

    edge_properties = edges[0][3]

    # Should have basic metadata but no weights
    assert "source_node_id" in edge_properties
    assert "target_node_id" in edge_properties
    assert "relationship_name" in edge_properties
    assert "updated_at" in edge_properties
    assert edge_properties["relationship_name"] == "purchased"

    # Should not have weight fields
    assert "weight" not in edge_properties
    weight_fields = [k for k in edge_properties.keys() if k.startswith("weight_")]
    assert len(weight_fields) == 0, f"Should have no weight fields, but found: {weight_fields}"
