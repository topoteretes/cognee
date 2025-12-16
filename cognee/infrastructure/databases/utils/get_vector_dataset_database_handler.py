from cognee.modules.users.models.DatasetDatabase import DatasetDatabase


def get_vector_dataset_database_handler(dataset_database: DatasetDatabase) -> dict:
    from cognee.infrastructure.databases.dataset_database_handler.supported_dataset_database_handlers import (
        supported_dataset_database_handlers,
    )

    handler = supported_dataset_database_handlers[dataset_database.vector_dataset_database_handler]
    return handler
