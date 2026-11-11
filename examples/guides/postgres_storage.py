"""Use the opt-in Postgres graph + pgvector batch writer.

Install matching patched cognee and cognee-postgres-graph versions, configure the
Postgres connection and embedding provider in your environment, and grant TEMP
on the vector database. The default database-per-dataset handlers also require
CREATE DATABASE. Run: python examples/guides/postgres_storage.py

VECTOR_WRITE_BATCH_SIZE=128 is independent of EMBEDDING_BATCH_SIZE. Four COPY
writers per collection overlap embedding and writes. The binary write pool adds
up to eight connections per adapter; VECTOR_POOL_ARGS controls the separate
SQLAlchemy pool. Measure COPY with your workload: the default binary writes can
be faster. Graph and vector transactions commit separately. A vector failure
does not roll back already committed graph rows or earlier vector batches.

COGNEE_STORAGE_TRACE=true emits payload-free per-batch timing. Keep CACHING=true
and DATASET_QUEUE_ENABLED=true when comparing performance.
"""

import asyncio
import os

os.environ.setdefault("GRAPH_DATABASE_PROVIDER", "postgres")
os.environ.setdefault("VECTOR_DB_PROVIDER", "pgvector")
os.environ.setdefault("POSTGRES_BATCH_WRITER_ENABLED", "true")
os.environ.setdefault("POSTGRES_VECTOR_WRITERS", "4")
os.environ.setdefault("VECTOR_WRITE_BATCH_SIZE", "128")

from cognee_postgres_graph import register

import cognee


async def main():
    # Registration uses register_batch_writer_factory in Cognee core. A plugin
    # factory must return None for unsupported adapters or a disabled opt-in.
    register()
    await cognee.add(
        "Pierre meets Prince Andrei in War and Peace.", dataset_name="postgres_example"
    )
    await cognee.cognify(datasets=["postgres_example"])


if __name__ == "__main__":
    asyncio.run(main())
