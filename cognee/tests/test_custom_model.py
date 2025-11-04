import os
import pathlib
import cognee
from cognee.low_level import DataPoint
from cognee.infrastructure.databases.graph import get_graph_engine

async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_custom_model")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_custom_model")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Define a custom graph model for programming languages.
    class FieldType(DataPoint):
        name: str = "Field"
        metadata: dict = {"index_fields": ["name"]}

    class Field(DataPoint):
        name: str
        is_type: FieldType
        metadata: dict = {"index_fields": ["name"]}

    class ProgrammingLanguageType(DataPoint):
        name: str = "Programming Language"
        metadata: dict = {"index_fields": ["name"]}

    class ProgrammingLanguage(DataPoint):
        name: str
        used_in: list[Field] = []
        is_type: ProgrammingLanguageType
        metadata: dict = {"index_fields": ["name"]}

    text = (
        "Python is an interpreted, high-level, general-purpose programming language. It was created by Guido van Rossum and first released in 1991. "
        + "Python is widely used in data analysis, web development, and machine learning."
    )

    await cognee.add(text)
    await cognee.cognify(graph_model=ProgrammingLanguage)

    
    await cognee.visualize_graph(destination_file_path="cognee/tests/test_custom_model.html")

    graph_engine = await get_graph_engine()
    
    graph_db_provider = os.getenv("GRAPH_DATABASE_PROVIDER", "kuzu").lower()
    
    # Query for Python entity and verify it exists with correct type
    python_found = False
    python_type = None
    
    if graph_db_provider in ["neo4j", "neptune", "neptune_analytics"]:
        query = """
        MATCH (n)
        WHERE n.name = 'Python'
        RETURN n.name as name, n.type as type
        """
        results = await graph_engine.query(query)
        if results:
            python_found = True
            python_type = results[0]["type"]
    
    elif graph_db_provider == "kuzu":
        query = """
        MATCH (n:Node)
        WHERE n.name = 'Python'
        RETURN n.name, n.type
        """
        results = await graph_engine.query(query)
        if results:
            python_found = True
            python_type = results[0][1]
    
    else:
        raise ValueError(f"Unsupported graph database provider: {graph_db_provider}")

    assert python_found, "Python entity was not extracted from the text"
    assert python_type == "ProgrammingLanguage", f"Python entity has incorrect type: {python_type}, expected: ProgrammingLanguage"
    
    # Query for entities that should NOT exist (Guido van Rossum and 1991)
    guido_found = False
    year_1991_found = False
    
    if graph_db_provider in ["neo4j", "neptune", "neptune_analytics"]:
        query = """
        MATCH (n)
        WHERE n.name IN ['Guido van Rossum', '1991']
        RETURN n.name as name
        """
        results = await graph_engine.query(query)
        for result in results:
            if result["name"] == "Guido van Rossum":
                guido_found = True
            elif result["name"] == "1991":
                year_1991_found = True
    
    elif graph_db_provider == "kuzu":
        query = """
        MATCH (n:Node)
        WHERE n.name IN ['Guido van Rossum', '1991']
        RETURN n.name
        """
        results = await graph_engine.query(query)
        for result in results:
            if result[0] == "Guido van Rossum":
                guido_found = True
            elif result[0] == "1991":
                year_1991_found = True
    
    else:
        raise ValueError(f"Unsupported graph database provider: {graph_db_provider}")

    
    assert not guido_found, "Guido van Rossum should not be extracted as it's not in the custom graph model"
    assert not year_1991_found, "1991 should not be extracted as it's not in the custom graph model"
    
    # Query for Field entities that might have been extracted (data analysis, web development, machine learning)
    field_entities = []
    
    if graph_db_provider in ["neo4j", "neptune", "neptune_analytics"]:
        query = """
        MATCH (n)
        WHERE n.type = 'Field'
        RETURN n.name as name
        """
        results = await graph_engine.query(query)
        field_entities = [r["name"] for r in results]
    
    elif graph_db_provider == "kuzu":
        query = """
        MATCH (n:Node)
        WHERE n.type = 'Field'
        RETURN n.name
        """
        results = await graph_engine.query(query)
        field_entities = [r[0] for r in results if r[0]]
    else:
        raise ValueError(f"Unsupported graph database provider: {graph_db_provider}")

    assert len(field_entities) > 0, f"No Field entities were extracted. Expected fields like 'data analysis', 'web development', 'machine learning' but got: {field_entities}"
    
    expected_fields = ["data analysis", "web development", "machine learning"]
    found_expected_fields = [f for f in expected_fields if any(f in field.lower() for field in field_entities)]
    
    assert len(found_expected_fields) > 0, f"None of the expected Field entities were found. Expected at least one of {expected_fields}, but got: {field_entities}"
    


if __name__ == "__main__":
    import asyncio

    asyncio.run(main(), debug=True)
