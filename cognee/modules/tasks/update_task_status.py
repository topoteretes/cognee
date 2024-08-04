from cognee.infrastructure.databases.relational import get_relational_engine

def update_task_status(data_id: str, status: str):
    db_engine = get_relational_engine()
    db_engine.insert_data("cognee.cognee", "cognee_task_status", [dict(data_id = data_id, status = status)])
