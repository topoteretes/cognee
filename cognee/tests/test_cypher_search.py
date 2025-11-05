import json
import os
import pathlib

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.relational import create_db_and_tables
from cognee.modules.search.types import SearchType


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_cypher_search")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_cypher_search")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await create_db_and_tables()

    graph_engine = await get_graph_engine()

    now = "2025-11-05 00:00:00"
    person_props = json.dumps({"name": "Alice"})
    project_props = json.dumps({"name": "Apollo"})

    await graph_engine.query(
        """
        CREATE (p:Node {
            id: 'person-1',
            name: 'Alice',
            type: 'Person',
            properties: $person_props,
            created_at: timestamp($now),
            updated_at: timestamp($now)
        })
        """,
        {"now": now, "person_props": person_props},
    )

    await graph_engine.query(
        """
        CREATE (p:Node {
            id: 'project-1',
            name: 'Apollo',
            type: 'Project',
            properties: $project_props,
            created_at: timestamp($now),
            updated_at: timestamp($now)
        })
        """,
        {"now": now, "project_props": project_props},
    )

    await graph_engine.query(
        """
        MATCH (person:Node {id: 'person-1'}), (project:Node {id: 'project-1'})
        MERGE (person)-[r:EDGE {relationship_name: 'WORKS_ON'}]->(project)
        ON CREATE SET
            r.created_at = timestamp($now),
            r.updated_at = timestamp($now),
            r.properties = '{}'
        ON MATCH SET
            r.updated_at = timestamp($now),
            r.properties = '{}'
        """,
        {"now": now},
    )

    multi_column_raw = await cognee.search(
        query_type=SearchType.CYPHER,
        query_text="""
            MATCH (p:Node {id: 'person-1'})-[:EDGE {relationship_name: 'WORKS_ON'}]->(proj:Node {id: 'project-1'})
            RETURN p.properties AS person_properties, proj.properties AS project_properties
        """,
    )
    assert isinstance(multi_column_raw, list)
    assert multi_column_raw, "Search returned no rows"

    assert len(multi_column_raw) == 1
    person_raw, project_raw = multi_column_raw[0]
    assert isinstance(person_raw, str)
    assert isinstance(project_raw, str)

    person_props_result = json.loads(person_raw)
    project_props_result = json.loads(project_raw)
    assert person_props_result.get("name") == "Alice"
    assert project_props_result.get("name") == "Apollo"

    single_column_raw = await cognee.search(
        query_type=SearchType.CYPHER,
        query_text="""
            MATCH (p:Node {id: 'person-1'})-[:EDGE {relationship_name: 'WORKS_ON'}]->(proj:Node {id: 'project-1'})
            RETURN DISTINCT proj.properties AS project_properties
        """,
    )
    assert isinstance(single_column_raw, list)
    assert single_column_raw, "Search returned no rows"

    assert len(single_column_raw) == 1
    (project_raw,) = single_column_raw[0]
    assert isinstance(project_raw, str)

    project_only_props = json.loads(project_raw)
    assert project_only_props.get("name") == "Apollo"

    context_only_raw = await cognee.search(
        query_type=SearchType.CYPHER,
        query_text="""
            MATCH (p:Node {id: 'person-1'})-[:EDGE {relationship_name: 'WORKS_ON'}]->(proj:Node {id: 'project-1'})
            RETURN DISTINCT proj.properties AS project_properties
        """,
        only_context=True,
    )
    assert isinstance(context_only_raw, list)
    assert context_only_raw, "Context search returned no rows"

    assert len(context_only_raw) == 1
    context_entry = context_only_raw[0]
    assert isinstance(context_entry, dict)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
