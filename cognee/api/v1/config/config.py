from cognee.infrastructure import infrastructure_config

class config():
    @staticmethod
    def system_root_directory(system_root_directory: str):
        infrastructure_config.set_config({
            "system_root_directory": system_root_directory
        })

    @staticmethod
    def data_root_directory(data_root_directory: str):
        infrastructure_config.set_config({
            "data_root_directory": data_root_directory
        })
