from typing import List, Dict
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.databases.relational.get_migration_relational_engine import get_migration_relational_engine
from sqlalchemy import text
from cognee.tasks.schema.models import DatabaseSchema, SchemaTable, SchemaRelationship
from cognee.infrastructure.databases.relational.create_relational_engine import create_relational_engine
from datetime import datetime

async def ingest_database_schema(
    database_config: Dict,
    schema_name: str = "default",
    max_sample_rows: int = 5,
    node_set: List[str] = ["database_schema"]
) -> Dict[str, List[DataPoint]|DataPoint]:
    """
    Ingest database schema with sample data into dedicated nodeset
    
    Args:
        database_config: Database connection configuration
        schema_name: Name identifier for this schema
        max_sample_rows: Maximum sample rows per table
        node_set: Target nodeset (default: ["database_schema"])
    
    Returns:
        List of created DataPoint objects
    """
    engine = create_relational_engine(
        db_path=database_config.get("db_path", ""),
        db_name=database_config.get("db_name", "cognee_db"),
        db_host=database_config.get("db_host"),
        db_port=database_config.get("db_port"),
        db_username=database_config.get("db_username"),
        db_password=database_config.get("db_password"),
        db_provider=database_config.get("db_provider", "sqlite"),
    )
    schema = await engine.extract_schema()
    tables={}
    sample_data={}
    schema_tables = []
    schema_relationships = []
    async with engine.engine.begin() as cursor:
        for table_name, details in schema.items():
            rows_result = await cursor.execute(text(f"SELECT * FROM {table_name} LIMIT {max_sample_rows}"))
            rows = [dict(zip([col["name"] for col in details["columns"]], row)) for row in rows_result.fetchall()]
            count_result = await cursor.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
            row_count_estimate = count_result.scalar()
            schema_table = SchemaTable(
                table_name=table_name,
                schema_name=schema_name,
                columns=details["columns"],
                primary_key=details.get("primary_key"),
                foreign_keys=details.get("foreign_keys", []),
                sample_rows=rows,
                row_count_estimate=row_count_estimate
            )
            schema_tables.append(schema_table)
            tables[table_name] = details
            sample_data[table_name] = rows
            
            for fk in details.get("foreign_keys",[]):
                relationship = SchemaRelationship(
                    source_table=table_name,
                    target_table=fk["ref_table"],
                    relationship_type=fk["type"],
                    source_column=fk["source_column"],
                    target_column=fk["target_column"]
                )
                schema_relationships.append(relationship)
    database_schema = DatabaseSchema(
        schema_name=schema_name,
        database_type=database_config.get("db_provider", "sqlite"),
        tables=tables,
        sample_data=sample_data,
        extraction_timestamp=datetime.utcnow()
    )
    
    return{
        "database_schema": database_schema,
        "schema_tables": schema_tables,
        "relationships": schema_relationships
    }