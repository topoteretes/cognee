from cognee.infrastructure.databases.relational import get_relational_engine

def create_task_status_table():
    db_engine = get_relational_engine()

    db_engine.create_table("cognee.cognee", "cognee_task_status", [
        dict(name = "data_id", type = "STRING"),
        dict(name = "status", type = "STRING"),
        dict(name = "created_at", type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ])
