from cognee.infrastructure.InfrastructureConfig import infrastructure_config

def update_task_status(data_id: str, status: str):
    db_engine = infrastructure_config.get_config("database_engine")
    db_engine.insert_data("cognee_task_status", [dict(data_id = data_id, status = status)])
