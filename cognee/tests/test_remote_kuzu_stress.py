import asyncio
import random
import time
from cognee.infrastructure.databases.graph.kuzu.remote_kuzu_adapter import RemoteKuzuAdapter
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.shared.logging_utils import get_logger

# Test configuration
BATCH_SIZE = 5000
NUM_BATCHES = 10
TOTAL_NODES = BATCH_SIZE * NUM_BATCHES
TOTAL_RELATIONSHIPS = TOTAL_NODES - 1

logger = get_logger()


async def create_node(adapter, node):
    query = (
        "CREATE (n:TestNode {"
        f"id: '{node['id']}', "
        f"name: '{node['name']}', "
        f"value: {node['value']}"
        "})"
    )
    await adapter.query(query)


async def create_relationship(adapter, source_id, target_id):
    query = (
        "MATCH (n1:TestNode {id: '" + str(source_id) + "'}), "
        "(n2:TestNode {id: '" + str(target_id) + "'}) "
        "CREATE (n1)-[r:CONNECTS_TO {weight: " + str(random.random()) + "}]->(n2)"
    )
    await adapter.query(query)


async def process_batch(adapter, start_id, batch_size):
    batch_start = time.time()
    batch_nodes = []

    # Prepare batch data
    logger.info(f"Preparing batch {start_id // batch_size + 1}/{NUM_BATCHES}...")
    for j in range(batch_size):
        node_id = start_id + j
        properties = {
            "id": str(node_id),
            "name": f"TestNode_{node_id}",
            "value": random.randint(1, 1000),
        }
        batch_nodes.append(properties)

    # Create nodes concurrently
    logger.info(
        f"Creating {batch_size} nodes for batch {start_id // batch_size + 1}/{NUM_BATCHES}..."
    )
    nodes_start = time.time()
    node_tasks = [create_node(adapter, node) for node in batch_nodes]
    await asyncio.gather(*node_tasks)
    nodes_time = time.time() - nodes_start

    # Create relationships concurrently
    logger.info(f"Creating relationships for batch {start_id // batch_size + 1}/{NUM_BATCHES}...")
    rels_start = time.time()
    rel_tasks = [
        create_relationship(adapter, batch_nodes[j]["id"], batch_nodes[j + 1]["id"])
        for j in range(len(batch_nodes) - 1)
    ]
    await asyncio.gather(*rel_tasks)
    rels_time = time.time() - rels_start

    batch_time = time.time() - batch_start
    logger.info(f"Batch {start_id // batch_size + 1}/{NUM_BATCHES} completed in {batch_time:.2f}s")
    logger.info(f"  - Nodes creation: {nodes_time:.2f}s")
    logger.info(f"  - Relationships creation: {rels_time:.2f}s")
    return batch_time


async def create_test_data(adapter, batch_size=BATCH_SIZE):
    tasks = []

    # Create tasks for each batch
    for i in range(0, TOTAL_NODES, batch_size):
        task = asyncio.create_task(process_batch(adapter, i, batch_size))
        tasks.append(task)

    # Wait for all batches to complete
    batch_times = await asyncio.gather(*tasks)
    return sum(batch_times)


async def main():
    config = get_graph_config()
    adapter = RemoteKuzuAdapter(
        config.graph_database_url, config.graph_database_username, config.graph_database_password
    )

    try:
        logger.info("=== Starting Kuzu Stress Test ===")
        logger.info(f"Configuration: {NUM_BATCHES} batches of {BATCH_SIZE} nodes each")
        logger.info(f"Total nodes to create: {TOTAL_NODES}")
        logger.info(f"Total relationships to create: {TOTAL_RELATIONSHIPS}")
        start_time = time.time()

        # Drop existing tables in correct order (relationships first, then nodes)
        logger.info("[1/5] Dropping existing tables...")
        await adapter.query("DROP TABLE IF EXISTS CONNECTS_TO")
        await adapter.query("DROP TABLE IF EXISTS TestNode")

        # Create node table
        logger.info("[2/5] Creating node table structure...")
        await adapter.query("""
        CREATE NODE TABLE TestNode (
            id STRING,
            name STRING,
            value INT64,
            PRIMARY KEY (id)
        )
        """)

        # Create relationship table
        logger.info("[3/5] Creating relationship table structure...")
        await adapter.query("""
        CREATE REL TABLE CONNECTS_TO (
            FROM TestNode TO TestNode,
            weight DOUBLE
        )
        """)

        # Clear existing test data
        logger.info("[4/5] Clearing existing test data...")
        await adapter.query("MATCH (n:TestNode) DETACH DELETE n")

        # Create new test data
        logger.info(
            f"[5/5] Creating test data ({NUM_BATCHES} concurrent batches of {BATCH_SIZE} nodes each)..."
        )
        total_batch_time = await create_test_data(adapter)

        end_time = time.time()
        total_duration = end_time - start_time

        # Verify the data
        logger.info("Verifying data...")
        result = await adapter.query("MATCH (n:TestNode) RETURN COUNT(n) as count")
        logger.info(f"Total nodes created: {result}")

        result = await adapter.query("MATCH ()-[r:CONNECTS_TO]->() RETURN COUNT(r) as count")
        logger.info(f"Total relationships created: {result}")

        logger.info("=== Test Summary ===")
        logger.info(f"Total batch processing time: {total_batch_time:.2f} seconds")
        logger.info(f"Total execution time: {total_duration:.2f} seconds")

    finally:
        await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())
