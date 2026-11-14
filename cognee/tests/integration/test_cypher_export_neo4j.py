"""Load exported Cypher into Neo4j and verify property names survive.

Requires the neo4j extra and TEST_NEO4J_URL pointing to a test database.
TEST_NEO4J_USERNAME and TEST_NEO4J_PASSWORD configure authentication.
Only nodes created by this test are removed.
"""

import os
from uuid import uuid4

import pytest

from cognee.modules.migration.formats import write_cypher

pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_NEO4J_URL"),
    reason="TEST_NEO4J_URL not set; skipping live Neo4j integration test",
)


def test_cypher_export_preserves_node_and_edge_properties(tmp_path):
    neo4j = pytest.importorskip("neo4j")
    username = os.getenv("TEST_NEO4J_USERNAME") or os.getenv("TEST_NEO4J_USER")
    auth = (username, os.getenv("TEST_NEO4J_PASSWORD")) if username else None
    source, target = str(uuid4()), str(uuid4())
    node_properties = {
        "type": "Entity",
        "first-name": "Alice",
        "first name": "Carol",
        "firstname": "Bob",
        "a`b": "quoted",
        "ab": "plain",
        "!!!": "punctuation",
        "prop": "literal",
        "123": "numeric",
        "i-d": "not-the-node-id",
    }
    edge_properties = {
        "edge-text": "hyphenated",
        "edge text": "spaced",
        "edgetext": "plain",
        "a`b": "quoted",
        "ab": "plain",
    }
    destination = tmp_path / "graph.cypher"
    write_cypher(
        [(source, node_properties), (target, {"name": "Berlin"})],
        [(source, target, "lives_in", edge_properties)],
        destination,
    )

    with (
        neo4j.GraphDatabase.driver(os.environ["TEST_NEO4J_URL"], auth=auth) as driver,
        driver.session() as session,
    ):
        try:
            for line in destination.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("//"):
                    session.run(line).consume()

            record = session.run(
                "MATCH (a:CogneeNode {id: $source})-[r:lives_in]->"
                "(b:CogneeNode {id: $target}) "
                "RETURN properties(a) AS node, properties(r) AS edge",
                source=source,
                target=target,
            ).single(strict=True)
            assert record["node"] == {**node_properties, "id": source}
            assert record["edge"] == edge_properties
        finally:
            session.run(
                "MATCH (n:CogneeNode) WHERE n.id IN $ids DETACH DELETE n",
                ids=[source, target],
            ).consume()
