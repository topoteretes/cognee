from typing import Optional
from uuid import UUID
from abc import ABC, abstractmethod

from cognee.modules.users.models.User import User


class DatasetDatabaseHandlerInterface(ABC):
    @classmethod
    @abstractmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """
        Return a dictionary with connection info for a graph or vector database for the given dataset.
        Function can auto handle deploying of the actual database if needed, but is not necessary.
        Only providing connection info is sufficient, this info will be mapped when trying to connect to the provided dataset in the future.
        Needed for Cognee multi-tenant/multi-user and backend access control support.

        Dictionary returned from this function will be used to create a DatasetDatabase row in the relational database.
        From which internal mapping of dataset -> database connection info will be done.

        Each dataset needs to map to a unique graph or vector database when backend access control is enabled to facilitate a separation of concern for data.

        Args:
            dataset_id: UUID of the dataset if needed by the database creation logic
            user: User object if needed by the database creation logic
        Returns:
            dict: Connection info for the created graph or vector database instance.
        """
        pass

    @classmethod
    @abstractmethod
    async def delete_dataset(cls, dataset_id: UUID, user: User) -> None:
        """
        Delete the graph or vector database for the given dataset.
        Function should auto handle deleting of the actual database or send a request to the proper service to delete/mark the database as not needed for the given dataset.
        Needed for maintaining a database for Cognee multi-tenant/multi-user and backend access control.

        Args:
            dataset_id: UUID of the dataset
            user: User object
        """
        pass
