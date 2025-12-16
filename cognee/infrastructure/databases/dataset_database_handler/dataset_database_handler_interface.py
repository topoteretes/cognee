from typing import Optional
from uuid import UUID
from abc import ABC, abstractmethod

from cognee.modules.users.models.User import User
from cognee.modules.users.models.DatasetDatabase import DatasetDatabase


class DatasetDatabaseHandlerInterface(ABC):
    @classmethod
    @abstractmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        """
        Return a dictionary with database connection/resolution info for a graph or vector database for the given dataset.
        Function can auto handle deploying of the actual database if needed, but is not necessary.
        Only providing connection info is sufficient, this info will be mapped when trying to connect to the provided dataset in the future.
        Needed for Cognee multi-tenant/multi-user and backend access control support.

        Dictionary returned from this function will be used to create a DatasetDatabase row in the relational database.
        From which internal mapping of dataset -> database connection info will be done.

        The returned dictionary is stored verbatim in the relational database and is later passed to
        resolve_dataset_connection_info() at connection time. For safe credential handling, prefer
        returning only references to secrets or role identifiers, not plaintext credentials.

        Each dataset needs to map to a unique graph or vector database when backend access control is enabled to facilitate a separation of concern for data.

        Args:
            dataset_id: UUID of the dataset if needed by the database creation logic
            user: User object if needed by the database creation logic
        Returns:
            dict: Connection info for the created graph or vector database instance.
        """
        pass

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        """
        Resolve runtime connection details for a dataset’s backing graph/vector database.
        Function is intended to be overwritten to implement custom logic for resolving connection info.

        This method is invoked right before the application opens a connection for a given dataset.
        It receives the DatasetDatabase row that was persisted when create_dataset() ran and must
        return a modified instance of DatasetDatabase with concrete connection parameters that the client/driver can use.
        Do not update these new DatasetDatabase values in the relational database to avoid storing secure credentials.

        In case of separate graph and vector database handlers, each handler should implement its own logic for resolving
        connection info and only change parameters related to its appropriate database, the resolution function will then
        be called one after another with the updated DatasetDatabase value from the previous function as the input.

        Typical behavior:
        - If the DatasetDatabase row already contains raw connection fields (e.g., host/port/db/user/password
        or api_url/api_key), return them as-is.
        - If the row stores only references (e.g., secret IDs, vault paths, cloud resource ARNs/IDs, IAM
        roles, SSO tokens), resolve those references by calling the appropriate secret manager or provider
        API to obtain short-lived credentials and assemble the final connection DatasetDatabase object.
        - Do not persist any resolved or decrypted secrets back to the relational database. Return them only
        to the caller.

        Args:
            dataset_database: DatasetDatabase row from the relational database
        Returns:
            DatasetDatabase: Updated instance with resolved connection info
        """
        return dataset_database

    @classmethod
    @abstractmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        """
        Delete the graph or vector database for the given dataset.
        Function should auto handle deleting of the actual database or send a request to the proper service to delete/mark the database as not needed for the given dataset.
        Needed for maintaining a database for Cognee multi-tenant/multi-user and backend access control.

        Args:
            dataset_database: DatasetDatabase row containing connection/resolution info for the graph or vector database to delete.
        """
        pass
