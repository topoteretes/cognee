from .supported_dataset_database_handlers import supported_dataset_database_handlers


def use_dataset_database_handler(
    dataset_database_handler_name, dataset_database_handler, dataset_database_provider
):
    supported_dataset_database_handlers[dataset_database_handler_name] = {
        "handler_instance": dataset_database_handler,
        "handler_provider": dataset_database_provider,
    }
