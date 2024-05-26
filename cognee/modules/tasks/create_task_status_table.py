from cognee.infrastructure.InfrastructureConfig import infrastructure_config
from cognee.infrastructure.databases.relational.config import get_relationaldb_config

config = get_relationaldb_config()

def create_task_status_table():
    db_engine = config.db_engine

    db_engine.create_table("cognee_task_status", [
        dict(name = "data_id", type = "STRING"),
        dict(name = "status", type = "STRING"),
        dict(name = "created_at", type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ])
