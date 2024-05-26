from cognee.infrastructure.InfrastructureConfig import infrastructure_config
from cognee.infrastructure.databases.relational.config import get_relationaldb_config

config = get_relationaldb_config()

def update_task_status(data_id: str, status: str):
    db_engine = config.db_engine
    db_engine.insert_data("cognee_task_status", [dict(data_id = data_id, status = status)])
