from cognee.infrastructure.databases.relational.config import get_relationaldb_config

def update_task_status(data_id: str, status: str):
    config = get_relationaldb_config()
    db_engine = config.database_engine
    db_engine.insert_data("cognee.cognee", "cognee_task_status", [dict(data_id = data_id, status = status)])
