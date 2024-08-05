from cognee.infrastructure.databases.relational import get_relational_engine

async def create_task_status_table():
    db_engine = get_relational_engine()

    await db_engine.create_table("cognee", "cognee_task_status", [
        dict(name="data_id", type="VARCHAR"),
        dict(name="status", type="VARCHAR"),
        dict(name="created_at", type="TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ])
