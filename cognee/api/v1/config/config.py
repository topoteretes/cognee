from typing import Optional
from cognee.infrastructure import infrastructure_config

class config():
    @staticmethod
    def data_path(data_path: Optional[str] = None) -> str:
        infrastructure_config.set_config({
            "data_path": data_path
        })
