from typing import List, Dict, Optional
from uuid import uuid5, NAMESPACE_OID
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.databases.relational.get_migration_relational_engine import (
    get_migration_relational_engine,
)
from sqlalchemy import text
from cognee.tasks.schema.models import DatabaseSchema, SchemaTable, SchemaRelationship
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from datetime import datetime


async def ingest_database_schema(
    database_config: Dict,
    schema_name: str = "default",
    max_sample_rows: int = 5,
) -> Dict[str, List[DataPoint] | DataPoint]:
    """
    Ingest database schema with sample data into dedicated nodeset

    Args:
        database_config: Database connection configuration
        schema_name: Name identifier for this schema
        max_sample_rows: Maximum sample rows per table

    Returns:
        List of created DataPoint objects
    """
    engine = create_relational_engine(
        db_path=database_config.get("migration_db_path", ""),
        db_name=database_config.get("migration_db_name", "cognee_db"),
        db_host=database_config.get("migration_db_host"),
        db_port=database_config.get("migration_db_port"),
        db_username=database_config.get("migration_db_username"),
        db_password=database_config.get("migration_db_password"),
        db_provider=database_config.get("migration_db_provider", "sqlite"),
    )
    schema = await engine.extract_schema()
    tables = {}
    sample_data = {}
    schema_tables = []
    schema_relationships = []

    async with engine.engine.begin() as cursor:
        for table_name, details in schema.items():
            qi = engine.engine.dialect.identifier_preparer.quote
            qname = lambda name : ".".join(qi(p) for p in name.split("."))
            tn = qname(table_name)
            tn = qi(table_name)
            rows_result = await cursor.execute(
                text(f"SELECT * FROM {tn} LIMIT :limit;"),
                {"limit": max_sample_rows}
            )
            rows = [
                dict(zip([col["name"] for col in details["columns"]], row))
                for row in rows_result.fetchall()
            ]
            count_result = await cursor.execute(text(f"SELECT COUNT(*) FROM {tn};"))
            row_count_estimate = count_result.scalar()

            schema_table = SchemaTable(
                id=uuid5(NAMESPACE_OID, name=f"{schema_name}:{tn}"),
                table_name=table_name,
                schema_name=schema_name,
                columns=details["columns"],
                primary_key=details.get("primary_key"),
                foreign_keys=details.get("foreign_keys", []),
                sample_rows=rows,
                row_count_estimate=row_count_estimate,
                description=f"Schema table for '{table_name}' with {len(details['columns'])} columns and approx. {row_count_estimate} rows.",
            )
            schema_tables.append(schema_table)
            tables[table_name] = details
            sample_data[table_name] = rows

            for fk in details.get("foreign_keys", []):
                ref_table_fq = fk["ref_table"]
                if '.' not in ref_table_fq and '.' in table_name:
                    ref_table_fq = f"{table_name.split('.', 1)[0]}.{ref_table_fq}"
                    
                relationship = SchemaRelationship(
                    id=uuid5(
                        NAMESPACE_OID,
                        name=f"{schema_name}:{table_name}:{fk['column']}->{ref_table_fq}:{fk['ref_column']}",
                    ),
                    source_table=table_name,
                    target_table=ref_table_fq,
                    relationship_type="foreign_key",
                    source_column=fk["column"],
                    target_column=fk["ref_column"],
                    description=f"Foreign key relationship: {table_name}.{fk['column']} → {ref_table_fq}.{fk['ref_column']}",
                )
                schema_relationships.append(relationship)

    database_schema = DatabaseSchema(
        id=uuid5(NAMESPACE_OID, name=schema_name),
        schema_name=schema_name,
        database_type=database_config.get("migration_db_provider", "sqlite"),
        tables=tables,
        sample_data=sample_data,
        extraction_timestamp=datetime.utcnow(),
        description=f"Database schema '{schema_name}' containing {len(schema_tables)} tables and {len(schema_relationships)} relationships.",
    )

    return {
        "database_schema": database_schema,
        "schema_tables": schema_tables,
        "relationships": schema_relationships,
    }
