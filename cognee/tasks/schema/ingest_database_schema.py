from typing import List, Dict
from uuid import uuid5, NAMESPACE_OID
from cognee.infrastructure.engine.models.DataPoint import DataPoint
from sqlalchemy import text
from cognee.tasks.schema.models import DatabaseSchema, SchemaTable, SchemaRelationship
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from datetime import datetime, timezone


async def ingest_database_schema(
    database_config: Dict,
    schema_name: str = "default",
    max_sample_rows: int = 0,
) -> Dict[str, List[DataPoint] | DataPoint]:
    """
    Extract database schema metadata (optionally with sample data) and return DataPoint models for graph construction.

    Args:
        database_config: Database connection configuration
        schema_name: Name identifier for this schema
        max_sample_rows: Maximum sample rows per table (0 means no sampling)

    Returns:
        Dict with keys:
            "database_schema": DatabaseSchema
            "schema_tables": List[SchemaTable]
            "relationships": List[SchemaRelationship]
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
            row_count_estimate = 0
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
                id=uuid5(NAMESPACE_OID, name=f"{schema_name}:{table_name}"),
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
                if "." not in ref_table_fq and "." in table_name:
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

    id_str = f"{database_config.get('migration_db_provider', 'sqlite')}:{database_config.get('migration_db_name', 'cognee_db')}:{schema_name}"
    database_schema = DatabaseSchema(
        id=uuid5(NAMESPACE_OID, name=id_str),
        schema_name=schema_name,
        database_type=database_config.get("migration_db_provider", "sqlite"),
        tables=tables,
        sample_data=sample_data,
        extraction_timestamp=datetime.now(timezone.utc),
        description=f"Database schema '{schema_name}' containing {len(schema_tables)} tables and {len(schema_relationships)} relationships.",
    )

    return {
        "database_schema": database_schema,
        "schema_tables": schema_tables,
        "relationships": schema_relationships,
    }
