from abc import ABC, abstractmethod
from typing import Any


class DataFetcherInterface(ABC):
    @abstractmethod
    def fetcher_name(self) -> str:
        pass

    @abstractmethod
    async def fetch(self, data_item_path: str, fetchers_config: dict[str, Any]) -> str:
        """
        args: data_item_path - path to the data item
        """
        pass
