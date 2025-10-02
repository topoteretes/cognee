import json
from typing import List, Dict
from uuid import uuid5, NAMESPACE_OID
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from sqlalchemy import text
from cognee.tasks.schema.models import DatabaseSchema, SchemaTable, SchemaRelationship
from cognee.infrastructure.databases.relational.get_migration_relational_engine import (
    get_migration_relational_engine,
)
from cognee.infrastructure.databases.relational.config import get_migration_config
from datetime import datetime, timezone


async def ingest_database_schema(
    schema,
    max_sample_rows: int = 0,
) -> Dict[str, List[DataPoint] | DataPoint]:
    """
    Extract database schema metadata (optionally with sample data) and return DataPoint models for graph construction.

    Args:
        schema: Database schema
        max_sample_rows: Maximum sample rows per table (0 means no sampling)

    Returns:
        Dict with keys:
            "database_schema": DatabaseSchema
            "schema_tables": List[SchemaTable]
            "relationships": List[SchemaRelationship]
    """

    tables = {}
    sample_data = {}
    schema_tables = []
    schema_relationships = []

    migration_config = get_migration_config()
    engine = get_migration_relational_engine()
    qi = engine.engine.dialect.identifier_preparer.quote
    try:
        max_sample_rows = max(0, int(max_sample_rows))
    except (TypeError, ValueError):
        max_sample_rows = 0

    def qname(name: str):
        split_name = name.split(".")
        return ".".join(qi(p) for p in split_name)

    async with engine.engine.begin() as cursor:
        for table_name, details in schema.items():
            tn = qname(table_name)
            if max_sample_rows > 0:
                rows_result = await cursor.execute(
                    text(f"SELECT * FROM {tn} LIMIT :limit;"),  # noqa: S608 - tn is fully quoted
                    {"limit": max_sample_rows},
                )
                rows = [dict(r) for r in rows_result.mappings().all()]
            else:
                rows = []

            if engine.engine.dialect.name == "postgresql":
                if "." in table_name:
                    schema_part, table_part = table_name.split(".", 1)
                else:
                    schema_part, table_part = "public", table_name
                estimate = await cursor.execute(
                    text(
                        "SELECT reltuples::bigint AS estimate "
                        "FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = :schema AND c.relname = :table"
                    ),
                    {"schema": schema_part, "table": table_part},
                )
                row_count_estimate = estimate.scalar() or 0
            else:
                count_result = await cursor.execute(text(f"SELECT COUNT(*) FROM {tn};"))  # noqa: S608 - tn is fully quoted
                row_count_estimate = count_result.scalar()

            schema_table = SchemaTable(
                id=uuid5(NAMESPACE_OID, name=f"{table_name}"),
                name=table_name,
                columns=json.dumps(details["columns"], default=str),
                primary_key=details.get("primary_key"),
                foreign_keys=json.dumps(details.get("foreign_keys", []), default=str),
                sample_rows=json.dumps(rows, default=str),
                row_count_estimate=row_count_estimate,
                description=f"Relational database table with '{table_name}' with {len(details['columns'])} columns and approx. {row_count_estimate} rows."
                f"Here are the columns this table contains: {details['columns']}"
                f"Here are a few sample_rows to show the contents of the table: {rows}"
                f"Table is part of the database: {migration_config.migration_db_name}",
            )
            schema_tables.append(schema_table)
            tables[table_name] = details
            sample_data[table_name] = rows

            for fk in details.get("foreign_keys", []):
                ref_table_fq = fk["ref_table"]
                if "." not in ref_table_fq and "." in table_name:
                    ref_table_fq = f"{table_name.split('.', 1)[0]}.{ref_table_fq}"

                relationship_name = (
                    f"{table_name}:{fk['column']}->{ref_table_fq}:{fk['ref_column']}"
                )
                relationship = SchemaRelationship(
                    id=uuid5(NAMESPACE_OID, name=relationship_name),
                    name=relationship_name,
                    source_table=table_name,
                    target_table=ref_table_fq,
                    relationship_type="foreign_key",
                    source_column=fk["column"],
                    target_column=fk["ref_column"],
                    description=f"Relational database table foreign key relationship between: {table_name}.{fk['column']} → {ref_table_fq}.{fk['ref_column']}"
                    f"This foreing key relationship between table columns is a part of the following database: {migration_config.migration_db_name}",
                )
                schema_relationships.append(relationship)

    id_str = f"{migration_config.migration_db_provider}:{migration_config.migration_db_name}"
    database_schema = DatabaseSchema(
        id=uuid5(NAMESPACE_OID, name=id_str),
        name=migration_config.migration_db_name,
        database_type=migration_config.migration_db_provider,
        tables=json.dumps(tables, default=str),
        sample_data=json.dumps(sample_data, default=str),
        description=f"Database schema containing {len(schema_tables)} tables and {len(schema_relationships)} relationships. "
        f"The database type is {migration_config.migration_db_provider}."
        f"The database contains the following tables: {tables}",
    )

    return {
        "database_schema": database_schema,
        "schema_tables": schema_tables,
        "relationships": schema_relationships,
    }
