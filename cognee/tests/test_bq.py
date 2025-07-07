import pytest
# from cognee.infrastructure.engine import DataPoint


async def main():

    print ("test")

    VECTOR_NODE_IDENTIFIER = "COGNEE_VECTOR_NODE"
    COLLECITON_PREFIX = "VECTOR_COLLECTION_"
    collection_name = "test"
    node_id = "node_id"

    nodes = ["node_1", "node_2"]
    embedding = [1.05, 3.14]
    #
    # result = (f"MATCH (n"
    #                             f":{VECTOR_NODE_IDENTIFIER} "
    #                             f":{COLLECITON_PREFIX}{collection_name}) "
    #                             f"WHERE id(n) IN {nodes}"
    #                             f"DETACH DELETE n")


    result = (f"MERGE (n"
                f":{VECTOR_NODE_IDENTIFIER} "
                f":{COLLECITON_PREFIX}{collection_name}) "
                f"{{~id: '{node_id}'}} "
                f"WITH n "
                f"CALL neptune.algo.vectors.upsert('{node_id}', {embedding}) "
                f"YIELD success "
                f"RETURN success ")
    print(result)
    #
    # data = DataPoint()

    # print(data)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
